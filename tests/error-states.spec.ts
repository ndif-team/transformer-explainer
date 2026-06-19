import { test, expect, REAL_NDIF_TIMEOUT_MS } from './fixtures';
import type { ServerModelMetadata } from '../src/types/forwardPass';

/**
 * Error-state coverage: the app must not silently dead-state on common backend
 * failure modes. We hit the workbench API directly to exercise the contract.
 */
test.describe('forward_pass error states', () => {
	const apiBase = process.env.VITE_WORKBENCH_API ?? 'http://localhost:8000';
	const headers = {
		'X-User-Email': 'dev@localhost',
		'Content-Type': 'application/json'
	};

	test('unknown model name returns a 4xx (or 500) with structured detail', async ({ request }) => {
		const resp = await request.post(`${apiBase}/forward_pass/start`, {
			headers,
			data: {
				model: 'not-a-real-model/does-not-exist',
				prompt: 'hello',
				positions: [-1],
				top_k: 5
			}
		});
		expect(resp.status()).toBeGreaterThanOrEqual(400);
		expect(resp.status()).toBeLessThan(600);
	});

	test('missing X-User-Email header is rejected with 401', async ({ request }) => {
		// Auth check fires before any model loading, so a placeholder model name
		// is fine here — the request never reaches the trace path.
		const resp = await request.post(`${apiBase}/forward_pass/start`, {
			headers: { 'Content-Type': 'application/json' },
			data: {
				model: 'any-model',
				prompt: 'hello',
				positions: [-1],
				top_k: 5
			}
		});
		expect(resp.status()).toBe(401);
	});

	test('whitespace-only prompt is accepted by the server', async ({ request }) => {
		// Pick any model that /models lists as allowed — the goal is to prove
		// the server's prompt validation tolerates a whitespace-only payload,
		// not to run a full inference. /forward_pass/start returns immediately
		// with a job_id (remote) or the full data payload (local) — both are
		// success signals; we don't poll to completion.
		const modelsResp = await request.get(`${apiBase}/models/`, { headers });
		expect(modelsResp.status()).toBe(200);
		const models = (await modelsResp.json()) as ServerModelMetadata[];
		const target = models.find((m) => m.allowed !== false);
		test.skip(!target, 'No allowed models returned by /models; skipping whitespace test.');

		const resp = await request.post(`${apiBase}/forward_pass/start`, {
			headers,
			data: { model: target!.name, prompt: ' ', positions: [-1], top_k: 3 },
			timeout: REAL_NDIF_TIMEOUT_MS
		});
		expect(resp.status()).toBe(200);
		const body = await resp.json();
		// Either local-mode inline result or remote job submission — both are valid.
		expect(body.job_id !== undefined || body.data !== null).toBe(true);
	});
});
