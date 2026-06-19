import { test, expect, REAL_NDIF_TIMEOUT_MS } from './fixtures';
import type { ForwardPassData, ServerModelMetadata } from '../src/types/forwardPass';
import type { APIRequestContext } from '@playwright/test';

/**
 * End-to-end coverage for Llama-3.1-8B on real NDIF, driven through workbench's
 * /forward_pass endpoint (start → status proxy → results).
 *
 * Two test cases, both against the same model:
 *   • short-prompt: 24 tokens, exhaustive cell-level assertions
 *     (every head × every q-row row sums, every (h,q,k) mask cell).
 *   • long-prompt: ≥128 tokens, sampled assertions
 *     (attention is O(S² · L · H) so payload grows ~16× from 24→128;
 *     iterating every cell in JS at that size is slow and the sampling
 *     pattern still catches RoPE/GQA/mask regressions).
 *
 * Skips both with a clear reason if Llama-3.1-8B isn't HOT on NDIF right now.
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
// doesn't flake. Composed of two complete sentences about the transformer
// architecture so the model lands on a high-confidence content word.
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

/** start → poll → results dance, parameterized by per-job timeout. */
async function runForwardPass(
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
	expect(
		finalStatus,
		`NDIF job ${job_id} did not complete (last description=${lastDescription ?? 'null'})`
	).toBe('COMPLETED');

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
 * Long-prompt variant: returns the raw Buffer instead of the parsed payload.
 * At S≥128 the decoded JSON (~600+ MB) exceeds V8's max string size
 * (0x1fffffe8 ≈ 536 MB), so .json() can't materialize it. The caller must
 * extract small slices via extractJsonValue() and parse those individually.
 */
async function runForwardPassRaw(
	request: APIRequestContext,
	prompt: string,
	pollTimeoutMs: number
): Promise<Buffer> {
	const body = { model: TARGET, prompt, positions: [-1], top_k: 10 };

	const startResp = await request.post(`${API_BASE}/forward_pass/start`, {
		headers: HEADERS,
		data: body,
		timeout: pollTimeoutMs
	});
	expect(startResp.status()).toBe(200);
	const { job_id } = (await startResp.json()) as { job_id?: string };
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
	expect(
		finalStatus,
		`NDIF job ${job_id} did not complete (last description=${lastDescription ?? 'null'})`
	).toBe('COMPLETED');

	const resultsResp = await request.post(`${API_BASE}/forward_pass/results/${job_id}`, {
		headers: HEADERS,
		data: body,
		timeout: pollTimeoutMs
	});
	expect(resultsResp.status()).toBe(200);
	return resultsResp.body();
}

/**
 * Find the matching closing bracket/brace, respecting strings and escapes.
 * `start` is the index of the opening bracket; returns the index of the
 * matching close. Used to slice individual JSON values out of a large Buffer
 * without forcing the whole thing through V8's string max.
 */
function findMatching(buf: Buffer, start: number, open: number, close: number): number {
	let depth = 0;
	let inString = false;
	for (let i = start; i < buf.length; i++) {
		const c = buf[i];
		if (inString) {
			if (c === 0x5c) i++; // skip escaped char
			else if (c === 0x22) inString = false; // "
		} else {
			if (c === 0x22) inString = true;
			else if (c === open) depth++;
			else if (c === close) {
				depth--;
				if (depth === 0) return i;
			}
		}
	}
	throw new Error('findMatching: unmatched bracket');
}

/**
 * Extract a top-level JSON value at `"<key>":` from a Buffer and JSON.parse it.
 * Only finds the FIRST occurrence — fine for the response shape we have where
 * each key appears once at the top level (under "data").
 */
function extractJsonValue(buf: Buffer, key: string): unknown {
	const marker = Buffer.from(`"${key}":`);
	const at = buf.indexOf(marker);
	if (at < 0) throw new Error(`key "${key}" not found in response body`);
	let i = at + marker.length;
	while (i < buf.length && (buf[i] === 0x20 || buf[i] === 0x09)) i++;
	const first = buf[i];
	let end: number;
	if (first === 0x7b) end = findMatching(buf, i, 0x7b, 0x7d); // {
	else if (first === 0x5b) end = findMatching(buf, i, 0x5b, 0x5d); // [
	else if (first === 0x22) {
		// string: scan to closing "
		let j = i + 1;
		while (j < buf.length) {
			if (buf[j] === 0x5c) j += 2;
			else if (buf[j] === 0x22) {
				end = j;
				break;
			} else j++;
		}
		end = end!;
	} else {
		// scalar (number/bool/null)
		let j = i;
		while (j < buf.length && buf[j] !== 0x2c && buf[j] !== 0x7d && buf[j] !== 0x5d) j++;
		end = j - 1;
	}
	return JSON.parse(buf.subarray(i, end + 1).toString('utf-8'));
}

/**
 * Extract the FIRST element of the "layers" array from the response Buffer.
 * Returns the parsed LayerPayload. At S=128 this is ~10 MB JSON — well
 * under V8's string max — even though the full layers array (~580 MB) is not.
 */
function extractLayer0(buf: Buffer): ForwardPassData['layers'][number] {
	const marker = Buffer.from('"layers":[');
	const at = buf.indexOf(marker);
	if (at < 0) throw new Error('"layers" not found in response body');
	// Skip "layers":[ then any whitespace to first {
	let i = at + marker.length;
	while (i < buf.length && (buf[i] === 0x20 || buf[i] === 0x09 || buf[i] === 0x0a)) i++;
	if (buf[i] !== 0x7b) throw new Error('layers[0] does not start with `{`');
	const end = findMatching(buf, i, 0x7b, 0x7d);
	return JSON.parse(buf.subarray(i, end + 1).toString('utf-8'));
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
		// /models's own derivation of the arch shape should already agree.
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

		// Layer count + per-layer attention shape.
		expect(payload.layers).toHaveLength(32);
		const S = payload.input_tokens.length;
		for (const layerIdx of [0, 15, 31]) {
			const probs = payload.layers[layerIdx].attention.probs;
			expect(probs, `layer ${layerIdx} probs heads`).toHaveLength(32);
			expect(probs[0]).toHaveLength(S);
			expect(probs[0][0]).toHaveLength(S);
		}

		// Causal-masked softmax invariant: every (head × q-row) row sums to ≈ 1.
		// 4-decimal rounding in to_data_obj caps the tolerance window.
		const layer0Probs = payload.layers[0].attention.probs;
		for (let h = 0; h < layer0Probs.length; h++) {
			for (let q = 0; q < S; q++) {
				const rowSum = layer0Probs[h][q].reduce((a, b) => a + b, 0);
				expect(rowSum, `layer 0 head ${h} q=${q} row sum`).toBeGreaterThan(0.98);
				expect(rowSum, `layer 0 head ${h} q=${q} row sum`).toBeLessThan(1.02);
			}
		}

		// Causal mask: upper triangle is exactly 0.
		for (const h of [0, 7, 15, 31]) {
			for (let q = 0; q < S; q++) {
				for (let k = q + 1; k < S; k++) {
					expect(
						layer0Probs[h][q][k],
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
		// At S≈147 the response is ~93 MB gzipped (~760 MB decoded JSON) and the
		// dominant cost is the download itself (~2 min over loopback). Per-job
		// poll budget bumped to 4 min, overall test budget 5 min, so a slower
		// NDIF queue won't false-fail this case.
		const LONG_POLL_MS = 240_000;
		test.setTimeout(LONG_POLL_MS + 60_000);

		// At S≥128 the full JSON exceeds V8's max string size, so we cannot
		// call .json() on the response. Read the body as a Buffer and pull out
		// the small sub-objects we actually need to assert against.
		const buf = await runForwardPassRaw(request, LONG_PROMPT, LONG_POLL_MS);

		// Sanity: a Buffer of the expected magnitude actually arrived. At
		// S=128, the decoded JSON is ~550 MB; at S≈147 (our prompt's actual
		// tokenization) it's ~760 MB. Floor at 200 MB so a regression that
		// dropped the bulk of layers/attention would fail loudly.
		expect(buf.length, 'response body suspiciously small').toBeGreaterThan(200_000_000);

		// data is the outer envelope; extractJsonValue walks past it because
		// it just searches for the first `"<key>":` and these keys only appear
		// inside data. Pull meta + input_tokens out of the leading portion.
		const meta = extractJsonValue(buf, 'meta') as ForwardPassData['meta'];
		assertLlama8bArch(meta.arch);

		const inputTokens = extractJsonValue(buf, 'input_tokens') as string[];
		const inputTokenIds = extractJsonValue(buf, 'input_token_ids') as number[];
		expect(
			inputTokens.length,
			`expected ≥128 tokens for long-prompt test; got ${inputTokens.length}`
		).toBeGreaterThanOrEqual(128);
		expect(inputTokens.length).toBe(inputTokenIds.length);
		const S = inputTokens.length;

		// Pull layer 0 out of the buffer. At S=128 this slice is ~10 MB —
		// well under V8's string max — so JSON.parse handles it fine.
		const layer0 = extractLayer0(buf);

		// Attention shape: 32 query heads (post-GQA-repeat), S×S per head.
		const probs = layer0.attention.probs;
		expect(probs).toHaveLength(32);
		expect(probs[0]).toHaveLength(S);
		expect(probs[0][0]).toHaveLength(S);

		// Sampled row-sum invariant: softmax over each q-row sums to ≈ 1
		// (4-decimal rounding caps tolerance). Sample a sparse grid of
		// (head, q) pairs covering head boundaries and sequence boundaries.
		const sampleHeads = [0, 7, 15, 23, 31];
		const sampleQs = [0, 1, Math.floor(S / 2), S - 2, S - 1];
		for (const h of sampleHeads) {
			for (const q of sampleQs) {
				const rowSum = probs[h][q].reduce((a, b) => a + b, 0);
				expect(rowSum, `layer 0 head ${h} q=${q} row sum`).toBeGreaterThan(0.98);
				expect(rowSum, `layer 0 head ${h} q=${q} row sum`).toBeLessThan(1.02);
			}
		}

		// Sampled causal-mask check: upper-triangle cells must be exactly 0.
		for (const h of sampleHeads) {
			expect(probs[h][0][1], `causal mask violated at layer 0 head ${h} q=0 k=1`).toBe(0);
			expect(
				probs[h][Math.floor(S / 2)][S - 1],
				`causal mask violated at layer 0 head ${h} q=S/2 k=S-1`
			).toBe(0);
		}

		// GQA storage: K/V at n_kv_heads=8 (not silently replicated to 32).
		const pp = layer0.per_position;
		expect(pp.q).toHaveLength(1);
		expect(pp.q[0]).toHaveLength(32);
		expect(pp.k[0]).toHaveLength(8);
		expect(pp.v[0]).toHaveLength(8);
		expect(pp.q[0][0]).toHaveLength(128);
		expect(pp.resid_pre[0]).toHaveLength(4096);

		// next_token sits near the end of the response; small object, safe
		// to extract directly.
		const nextToken = extractJsonValue(buf, 'next_token') as ForwardPassData['next_token'];
		expect(nextToken.tokens.length).toBeGreaterThanOrEqual(10);
		expect(nextToken.probs[0]).toBeGreaterThan(nextToken.probs.at(-1) ?? 1);
		expect(nextToken.probs[0]).toBeGreaterThan(0.05);
	});
});
