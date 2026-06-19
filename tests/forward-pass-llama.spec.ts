import { test, expect, REAL_NDIF_TIMEOUT_MS } from './fixtures';
import type { ForwardPassData, ServerModelMetadata } from '../src/types/forwardPass';

/**
 * End-to-end coverage for Llama-3.1-8B on real NDIF, driven through workbench's
 * /forward_pass endpoint (start → status proxy → results). A long-sequence
 * prompt (>12 tokens) must succeed, and the payload must satisfy Llama-3.1-8B
 * shape invariants: 32 layers, 32 query heads, 8 KV heads (GQA), head_dim=128,
 * RoPE. The per-head attention-probs row sums and the upper-triangle zero
 * pattern prove RoPE + GQA + causal masking are wired correctly inside the
 * trace (not just type-shaped).
 *
 * Skips with a clear reason if Llama-3.1-8B isn't HOT on NDIF right now.
 */

const TARGET = 'meta-llama/Llama-3.1-8B';
// Crafted to tokenize to >12 tokens under Llama-3 tokenizer (locally measured
// at 24 tokens including BOS). Picks up a strong next-token signal too so the
// top-k assertion isn't flaky.
const LONG_PROMPT =
	'Once upon a time, in a small village nestled between two great mountains, ' +
	'there lived a young apprentice who dreamed of';

const API_BASE = (process.env.VITE_WORKBENCH_API ?? 'http://localhost:8000').replace(/\/$/, '');
const HEADERS = { 'X-User-Email': 'dev@localhost', 'Content-Type': 'application/json' };

test.describe('forward_pass Llama-3.1-8B', () => {
	test('long-sequence forward pass returns a Llama-3.1-8B shaped GQA+RoPE payload', async ({
		request
	}) => {
		test.setTimeout(REAL_NDIF_TIMEOUT_MS * 2);

		// /models must list the 8B and mark it allowed.
		const modelsResp = await request.get(`${API_BASE}/models/`, { headers: HEADERS });
		expect(modelsResp.status()).toBe(200);
		const models = (await modelsResp.json()) as ServerModelMetadata[];
		const target = models.find((m) => m.name === TARGET);
		test.skip(
			!target || target.allowed === false,
			`Llama-3.1-8B not HOT on NDIF right now (or not allowed for this key).`
		);
		// Cheap sanity: /models also knows the GQA shape.
		expect(target!.n_layers).toBe(32);
		expect(target!.n_heads).toBe(32);
		expect(target!.n_kv_heads).toBe(8);

		// Kick off the forward pass.
		const body = {
			model: TARGET,
			prompt: LONG_PROMPT,
			positions: [-1],
			top_k: 10
		};
		const startResp = await request.post(`${API_BASE}/forward_pass/start`, {
			headers: HEADERS,
			data: body
		});
		expect(startResp.status()).toBe(200);
		const { job_id, data: inlineData } = (await startResp.json()) as {
			job_id?: string;
			data?: ForwardPassData;
		};

		let payload: ForwardPassData;
		if (inlineData) {
			// Local-mode short-circuit (state.remote=false). Shouldn't happen against NDIF.
			payload = inlineData;
		} else {
			expect(job_id, '/forward_pass/start must return a NDIF job_id').toBeTruthy();
			const deadline = Date.now() + REAL_NDIF_TIMEOUT_MS;
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
				data: body
			});
			expect(resultsResp.status()).toBe(200);
			const result = (await resultsResp.json()) as { data: ForwardPassData };
			payload = result.data;
		}

		// --- arch shape: Llama-3.1-8B is fully specified ---
		const arch = payload.meta.arch;
		expect(arch.kind).toBe('llama');
		expect(arch.positional_kind).toBe('rope');
		expect(arch.has_fused_qkv).toBe(false);
		expect(arch.n_layers).toBe(32);
		expect(arch.n_heads).toBe(32);
		expect(arch.n_kv_heads).toBe(8); // GQA: 4 Q heads per KV head
		expect(arch.d_head).toBe(128);
		expect(arch.d_model).toBe(4096);
		expect(arch.vocab_size).toBe(128256);

		// --- long-sequence floor (the explicit user requirement) ---
		expect(
			payload.input_tokens.length,
			`expected >12 tokens for this prompt; got ${payload.input_tokens.length}`
		).toBeGreaterThan(12);
		expect(payload.input_tokens.length).toBe(payload.input_token_ids.length);

		// --- layer count + per-layer attention shape ---
		expect(payload.layers).toHaveLength(32);
		const S = payload.input_tokens.length;
		for (const layerIdx of [0, 15, 31]) {
			const probs = payload.layers[layerIdx].attention.probs;
			expect(probs, `layer ${layerIdx} probs heads`).toHaveLength(32); // post-GQA-repeat
			expect(probs[0]).toHaveLength(S);
			expect(probs[0][0]).toHaveLength(S);
		}

		// --- causal-masked softmax invariant: every row of probs sums to ≈ 1 ---
		// 4-decimal rounding in to_data_obj caps the tolerance window.
		const layer0Probs = payload.layers[0].attention.probs;
		for (let h = 0; h < layer0Probs.length; h++) {
			for (let q = 0; q < S; q++) {
				const rowSum = layer0Probs[h][q].reduce((a, b) => a + b, 0);
				expect(rowSum, `layer 0 head ${h} q=${q} row sum`).toBeGreaterThan(0.98);
				expect(rowSum, `layer 0 head ${h} q=${q} row sum`).toBeLessThan(1.02);
			}
		}

		// --- causal mask: upper triangle of probs is exactly 0 ---
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

		// --- GQA storage: K is stored with n_kv_heads, Q with n_heads ---
		// per_position shape: [pos][head][d_head]. Only 1 position (last) by request.
		const pp = payload.layers[0].per_position;
		expect(pp.q).toHaveLength(1);
		expect(pp.q[0]).toHaveLength(32);
		expect(pp.q[0][0]).toHaveLength(128);
		expect(pp.k[0]).toHaveLength(8); // ← the GQA assertion: K storage is 8, not 32
		expect(pp.v[0]).toHaveLength(8);

		// --- residual stream shape ---
		expect(pp.resid_pre[0]).toHaveLength(4096);
		expect(pp.resid_post[0]).toHaveLength(4096);

		// --- next-token top-k: real probability mass on " becoming" ---
		expect(payload.next_token.tokens.length).toBeGreaterThanOrEqual(10);
		expect(payload.next_token.probs[0]).toBeGreaterThan(payload.next_token.probs.at(-1) ?? 1);
		// We don't pin the exact token to avoid coupling to model quirks; just check
		// the head of the distribution carries real signal (>5%).
		expect(payload.next_token.probs[0]).toBeGreaterThan(0.05);
	});

});
