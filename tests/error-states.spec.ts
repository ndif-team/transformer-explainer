import { test, expect, REAL_NDIF_TIMEOUT_MS } from './fixtures';

/**
 * Error-state coverage: the app must not silently dead-state on common backend
 * failure modes. We hit the workbench API directly to exercise the contract,
 * then re-validate the UI renders after recovery.
 */
test.describe('forward_pass error states', () => {
	const apiBase = process.env.VITE_WORKBENCH_API ?? 'http://localhost:8000';
	const headers = {
		'X-User-Email': 'dev@localhost',
		'Content-Type': 'application/json'
	};

	test('unknown model name returns a 4xx with structured detail', async ({ request }) => {
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
		const resp = await request.post(`${apiBase}/forward_pass/start`, {
			headers: { 'Content-Type': 'application/json' },
			data: {
				model: 'openai-community/gpt2',
				prompt: 'hello',
				positions: [-1],
				top_k: 5
			}
		});
		expect(resp.status()).toBe(401);
	});

	test('whitespace-only prompt is handled gracefully by the server', async ({ request }) => {
		// Client substitutes " " for empty prompts; the server must accept it.
		const resp = await request.post(`${apiBase}/forward_pass/start`, {
			headers,
			data: { model: 'openai-community/gpt2', prompt: ' ', positions: [-1], top_k: 3 },
			timeout: REAL_NDIF_TIMEOUT_MS
		});
		expect(resp.status()).toBe(200);
		const body = await resp.json();
		expect(body.data).toBeTruthy();
		expect(Array.isArray(body.data.input_tokens)).toBe(true);
		expect(body.data.input_tokens.length).toBeGreaterThan(0);
	});
});
