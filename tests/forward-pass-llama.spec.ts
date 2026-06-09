import { test, expect, REAL_NDIF_TIMEOUT_MS, waitForBootstrap } from './fixtures';

/**
 * Llama happy path. Skips entirely if no Llama-family model is present in the
 * workbench /models response (typical when the local config is used or no
 * Llama is HOT on NDIF).
 */
test.describe('forward_pass Llama happy path', () => {
	test('switching to a Llama model renders RoPE branch and arch metadata', async ({
		explainerPage: page,
		request
	}) => {
		const apiBase = process.env.VITE_WORKBENCH_API ?? 'http://localhost:8000';
		const modelsResp = await request.get(`${apiBase}/models/`, {
			headers: { 'X-User-Email': 'dev@localhost' }
		});
		expect(modelsResp.status()).toBe(200);
		const models = (await modelsResp.json()) as Array<{
			name: string;
			arch_kind: string;
			allowed?: boolean;
		}>;
		const llama = models.find((m) => m.arch_kind === 'llama' && m.allowed !== false);
		test.skip(!llama, 'No Llama-family model HOT on the backend; skipping Llama spec.');

		await page.goto('/');
		await waitForBootstrap(page);

		// Open the model selector and pick the Llama option.
		await page.getByTestId('model-selector-button').click();
		await page.getByTestId('model-selector-list').waitFor();
		const llamaOption = page.getByTestId('model-selector-option-llama').first();
		const forwardPassRequest = page.waitForResponse(
			(resp) => resp.url().includes('/forward_pass/start') && resp.status() < 500,
			{ timeout: REAL_NDIF_TIMEOUT_MS }
		);
		await llamaOption.click();
		const fpResp = await forwardPassRequest;
		expect(fpResp.status()).toBeLessThan(500);

		// Selector reflects the new arch
		await expect(page.getByTestId('model-selector-arch')).toHaveText('llama');

		// Embedding wrapper switches to RoPE positional kind
		await expect(page.getByTestId('embedding-step')).toHaveAttribute(
			'data-positional-kind',
			'rope'
		);
		await expect(page.getByTestId('embedding-step')).toHaveAttribute('data-arch-kind', 'llama');

		// Block strip + top-k still render
		await expect(page.getByTestId('block-strip')).toBeVisible();
		await expect(page.getByTestId('next-token-topk')).toBeVisible();
	});
});
