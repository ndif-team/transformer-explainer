import { test, expect, REAL_NDIF_TIMEOUT_MS } from './fixtures';
import type { ForwardPassData, ServerModelMetadata } from '../src/types/forwardPass';
import { applyCausalMaskZero, computeAttentionProbs } from '../src/utils/deriveAttention';
import type { APIRequestContext } from '@playwright/test';

/**
 * End-to-end coverage for Llama-3.1-8B on real NDIF, driven through workbench's
 * /forward_pass endpoint (start → status proxy → results).
 *
 * Two test cases against the same model:
 *   • short-prompt (~24 tokens): exhaustive cell-level assertions over every
 *     head × every q-row of layer 0.
 *   • long-prompt (≥128 tokens): sampled assertions over a sparse grid of
 *     (head, q) pairs — iterating every cell at S≈147 is wasted JS time.
 *
 * Both use deriveAttention helpers to compute the causal-masked + softmax
 * views from the wire `scores`. That's the same path the UI's adaptForwardPass
 * runs through, so a bug in the derivation would surface here too.
 *
 * Skips with a clear reason if Llama-3.1-8B isn't HOT on NDIF right now.
 */

const TARGET = 'meta-llama/Llama-3.1-8B';
const API_BASE = (process.env.VITE_WORKBENCH_API ?? 'http://localhost:8000').replace(/\/$/, '');
const HEADERS = { 'X-User-Email': 'dev@localhost', 'Content-Type': 'application/json' };

// Short prompt — tokenizes to 24 tokens with BOS under Llama-3 tokenizer.
const SHORT_PROMPT =
	'Once upon a time, in a small village nestled between two great mountains, ' +
	'there lived a young apprentice who dreamed of';

// Long prompt — tokenizes to ~147 tokens (well above the 128 floor) and
// produces a strong, stable next-token signal so the top-k spot check
// doesn't flake.
const LONG_PROMPT =
	'The transformer architecture, introduced in the 2017 paper Attention ' +
	'Is All You Need by Vaswani and colleagues, fundamentally reshaped how ' +
	'neural networks process sequential information. Unlike recurrent ' +
	'networks that march through tokens one at a time, transformers compute ' +
	'self-attention over the entire sequence in parallel, allowing every ' +
	'position to gather context from every other position in a single layer. ' +
	'This design unlocked the modern era of large language models, including ' +
	'GPT, Llama, Mistral, Claude, Gemini, and many others. The architecture ' +
	'remains the dominant choice for both research and production systems, ' +
	'powering applications from chat assistants to code completion to ' +
	'scientific discovery, and its core mechanism — scaled dot-product ' +
	'attention — is now considered one of the most influential ideas in';

/**
 * NDIF transient error patterns: failures matching these are retryable. The
 * "is not whitelisted" path in particular fires randomly today (the same exact
 * request will succeed on one attempt and fail on another) — it's an NDIF-side
 * race in the per-worker module allowlist, not a contract failure.
 */
const NDIF_TRANSIENT_PATTERNS = [/is not whitelisted/i];

function isTransientNdifError(description: string | null): boolean {
	if (!description) return false;
	return NDIF_TRANSIENT_PATTERNS.some((p) => p.test(description));
}

/** Submit one job and wait for it. Returns the parsed payload or throws. */
async function runForwardPassOnce(
	request: APIRequestContext,
	prompt: string,
	pollTimeoutMs: number
): Promise<ForwardPassData> {
	const body = { model: TARGET, prompt, positions: [-1], top_k: 10 };

	const startResp = await request.post(`${API_BASE}/forward_pass/start`, {
		headers: HEADERS,
		data: body,
		timeout: pollTimeoutMs
	});
	expect(startResp.status()).toBe(200);
	const { job_id, data: inlineData } = (await startResp.json()) as {
		job_id?: string;
		data?: ForwardPassData;
	};

	if (inlineData) return inlineData; // local-mode short-circuit

	expect(job_id, '/forward_pass/start must return a NDIF job_id').toBeTruthy();
	const deadline = Date.now() + pollTimeoutMs;
	let finalStatus = '';
	let lastDescription: string | null = null;
	while (Date.now() < deadline) {
		const r = await request.get(`${API_BASE}/forward_pass/status/${job_id}`, {
			headers: HEADERS
		});
		expect(r.status()).toBe(200);
		const s = (await r.json()) as { status: string; description?: string | null };
		finalStatus = (s.status ?? '').toUpperCase();
		lastDescription = s.description ?? null;
		if (
			finalStatus === 'COMPLETED' ||
			finalStatus === 'ERROR' ||
			finalStatus === 'NNSIGHT_ERROR'
		)
			break;
		await new Promise((r) => setTimeout(r, 1000));
	}
	if (finalStatus !== 'COMPLETED') {
		const err = new Error(
			`NDIF job ${job_id} did not complete (status=${finalStatus}, description=${lastDescription ?? 'null'})`
		);
		(err as Error & { ndifDescription?: string | null }).ndifDescription = lastDescription;
		throw err;
	}

	const resultsResp = await request.post(`${API_BASE}/forward_pass/results/${job_id}`, {
		headers: HEADERS,
		data: body,
		timeout: pollTimeoutMs
	});
	expect(resultsResp.status()).toBe(200);
	const result = (await resultsResp.json()) as { data: ForwardPassData };
	return result.data;
}

/**
 * start → poll → results dance with up to N retries on known-transient NDIF
 * errors (currently the "not whitelisted" race that fires sporadically on
 * pinned models). Non-transient errors propagate on the first attempt.
 */
async function runForwardPass(
	request: APIRequestContext,
	prompt: string,
	pollTimeoutMs: number,
	// NDIF's "is not whitelisted" race currently fires for ~80% of jobs against
	// pinned Llama deployments — the same code path that succeeds reliably hours
	// later. 12 attempts at ~3 s per transient failure gives ~93% odds of one
	// completing per test while keeping the worst-case wall-clock bounded.
	maxAttempts = 12
): Promise<ForwardPassData> {
	let lastErr: Error | null = null;
	for (let attempt = 1; attempt <= maxAttempts; attempt++) {
		try {
			return await runForwardPassOnce(request, prompt, pollTimeoutMs);
		} catch (e) {
			const err = e as Error & { ndifDescription?: string | null };
			lastErr = err;
			if (!isTransientNdifError(err.ndifDescription ?? null) || attempt === maxAttempts) {
				throw err;
			}
			console.warn(
				`[retry ${attempt}/${maxAttempts}] NDIF transient error: ${err.ndifDescription}`
			);
			await new Promise((r) => setTimeout(r, 2000 * attempt));
		}
	}
	throw lastErr ?? new Error('runForwardPass: unreachable');
}

/** Common arch invariants for Llama-3.1-8B. */
function assertLlama8bArch(arch: ForwardPassData['meta']['arch']) {
	expect(arch.kind).toBe('llama');
	expect(arch.positional_kind).toBe('rope');
	expect(arch.has_fused_qkv).toBe(false);
	expect(arch.n_layers).toBe(32);
	expect(arch.n_heads).toBe(32);
	expect(arch.n_kv_heads).toBe(8); // GQA: 4 Q heads per KV head
	expect(arch.d_head).toBe(128);
	expect(arch.d_model).toBe(4096);
	expect(arch.vocab_size).toBe(128256);
}

test.describe('forward_pass Llama-3.1-8B', () => {
	test.beforeAll(async ({ request }) => {
		const modelsResp = await request.get(`${API_BASE}/models/`, { headers: HEADERS });
		expect(modelsResp.status()).toBe(200);
		const models = (await modelsResp.json()) as ServerModelMetadata[];
		const target = models.find((m) => m.name === TARGET);
		test.skip(
			!target || target.allowed === false,
			`Llama-3.1-8B not HOT on NDIF right now (or not allowed for this key).`
		);
		expect(target!.n_layers).toBe(32);
		expect(target!.n_heads).toBe(32);
		expect(target!.n_kv_heads).toBe(8);
	});

	test('short prompt (~24 tokens): exhaustive GQA+RoPE shape + invariants', async ({
		request
	}) => {
		test.setTimeout(REAL_NDIF_TIMEOUT_MS * 2);

		const payload = await runForwardPass(request, SHORT_PROMPT, REAL_NDIF_TIMEOUT_MS);
		assertLlama8bArch(payload.meta.arch);

		// Sequence length floor.
		expect(payload.input_tokens.length).toBeGreaterThan(12);
		expect(payload.input_tokens.length).toBe(payload.input_token_ids.length);

		// Layer count + per-layer attention shape (scores only — masked +
		// probs are derived client-side now).
		expect(payload.layers).toHaveLength(32);
		const S = payload.input_tokens.length;
		for (const layerIdx of [0, 15, 31]) {
			const scores = payload.layers[layerIdx].attention.scores;
			expect(scores, `layer ${layerIdx} scores heads`).toHaveLength(32);
			expect(scores[0]).toHaveLength(S);
			expect(scores[0][0]).toHaveLength(S);
		}

		// Derive probs the same way the UI's adapter does, then assert the
		// softmax invariant (every (head × q-row) row sums to ≈ 1).
		// This exercises both the server's scores math and the client-side
		// derivation in deriveAttention.ts.
		const layer0Scores = payload.layers[0].attention.scores;
		for (let h = 0; h < layer0Scores.length; h++) {
			const probsHead = computeAttentionProbs(layer0Scores[h]);
			for (let q = 0; q < S; q++) {
				const rowSum = probsHead[q].reduce((a, b) => a + b, 0);
				expect(rowSum, `layer 0 head ${h} q=${q} row sum`).toBeGreaterThan(0.98);
				expect(rowSum, `layer 0 head ${h} q=${q} row sum`).toBeLessThan(1.02);
			}
		}

		// Causal mask via applyCausalMaskZero: upper triangle is exactly 0.
		for (const h of [0, 7, 15, 31]) {
			const masked = applyCausalMaskZero(layer0Scores[h]);
			for (let q = 0; q < S; q++) {
				for (let k = q + 1; k < S; k++) {
					expect(
						masked[q][k],
						`causal mask violated at layer 0 head ${h} q=${q} k=${k}`
					).toBe(0);
				}
			}
		}

		// GQA storage: K/V at n_kv_heads (not silently replicated to n_heads).
		const pp = payload.layers[0].per_position;
		expect(pp.q).toHaveLength(1);
		expect(pp.q[0]).toHaveLength(32);
		expect(pp.q[0][0]).toHaveLength(128);
		expect(pp.k[0]).toHaveLength(8);
		expect(pp.v[0]).toHaveLength(8);

		// Residual width.
		expect(pp.resid_pre[0]).toHaveLength(4096);
		expect(pp.resid_post[0]).toHaveLength(4096);

		// Top-k carries real signal.
		expect(payload.next_token.tokens.length).toBeGreaterThanOrEqual(10);
		expect(payload.next_token.probs[0]).toBeGreaterThan(payload.next_token.probs.at(-1) ?? 1);
		expect(payload.next_token.probs[0]).toBeGreaterThan(0.05);
	});

	test('long prompt (≥128 tokens): payload survives, invariants hold on samples', async ({
		request
	}) => {
		// Server now only ships `scores` (no scores_masked / probs), so at
		// S≈147 the payload is ~290 MB decoded — comfortably under V8's
		// max string size (~536 MB). Plain .json() works again; no Buffer
		// extraction hack needed. Per-job poll budget bumped to 4 min;
		// overall test budget 5 min for slower NDIF queues.
		const LONG_POLL_MS = 240_000;
		test.setTimeout(LONG_POLL_MS + 60_000);

		const payload = await runForwardPass(request, LONG_PROMPT, LONG_POLL_MS);
		assertLlama8bArch(payload.meta.arch);

		// Sequence length floor — the actual point of this test.
		expect(
			payload.input_tokens.length,
			`expected ≥128 tokens for long-prompt test; got ${payload.input_tokens.length}`
		).toBeGreaterThanOrEqual(128);
		expect(payload.input_tokens.length).toBe(payload.input_token_ids.length);

		// Layer count + per-layer scores shape.
		expect(payload.layers).toHaveLength(32);
		const S = payload.input_tokens.length;
		for (const layerIdx of [0, 15, 31]) {
			const scores = payload.layers[layerIdx].attention.scores;
			expect(scores, `layer ${layerIdx} scores heads`).toHaveLength(32);
			expect(scores[0]).toHaveLength(S);
			expect(scores[0][0]).toHaveLength(S);
		}

		// Sampled invariants. Derive probs from a sparse grid of heads, then
		// check that each derived row sums to ≈ 1 and the mask zeros are in
		// the upper triangle. Compute the full probs matrix for a sampled
		// head once and read whichever cells we need.
		const layer0Scores = payload.layers[0].attention.scores;
		const sampleHeads = [0, 7, 15, 23, 31];
		const sampleQs = [0, 1, Math.floor(S / 2), S - 2, S - 1];
		for (const h of sampleHeads) {
			const probsHead = computeAttentionProbs(layer0Scores[h]);
			const maskedHead = applyCausalMaskZero(layer0Scores[h]);
			for (const q of sampleQs) {
				const rowSum = probsHead[q].reduce((a, b) => a + b, 0);
				expect(rowSum, `layer 0 head ${h} q=${q} row sum`).toBeGreaterThan(0.98);
				expect(rowSum, `layer 0 head ${h} q=${q} row sum`).toBeLessThan(1.02);
			}
			// Spot-check causal mask at two (q,k) pairs per head.
			expect(maskedHead[0][1], `mask violated at h=${h} q=0 k=1`).toBe(0);
			expect(
				maskedHead[Math.floor(S / 2)][S - 1],
				`mask violated at h=${h} q=S/2 k=S-1`
			).toBe(0);
		}

		// GQA storage: same shape as short-prompt; sequence length doesn't
		// change the per-position widths.
		const pp = payload.layers[0].per_position;
		expect(pp.q).toHaveLength(1);
		expect(pp.q[0]).toHaveLength(32);
		expect(pp.k[0]).toHaveLength(8);
		expect(pp.v[0]).toHaveLength(8);
		expect(pp.q[0][0]).toHaveLength(128);
		expect(pp.resid_pre[0]).toHaveLength(4096);

		// Top-k still carries real signal at the longer context.
		expect(payload.next_token.tokens.length).toBeGreaterThanOrEqual(10);
		expect(payload.next_token.probs[0]).toBeGreaterThan(payload.next_token.probs.at(-1) ?? 1);
		expect(payload.next_token.probs[0]).toBeGreaterThan(0.05);
	});
});
