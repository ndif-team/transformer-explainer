import { test, expect, REAL_NDIF_TIMEOUT_MS, waitForBootstrap } from './fixtures';

test.describe('forward_pass GPT-2 happy path', () => {
	test('app renders and reaches the workbench /forward_pass endpoint', async ({
		explainerPage: page
	}) => {
		await page.goto('/');
		await waitForBootstrap(page);

		const forwardPassRequest = page.waitForResponse(
			(resp) =>
				resp.url().includes('/forward_pass/') &&
				(resp.url().includes('/start') || resp.url().includes('/results/')) &&
				resp.status() < 500,
			{ timeout: REAL_NDIF_TIMEOUT_MS }
		);

		const resp = await forwardPassRequest;
		expect(resp.status()).toBeLessThan(500);

		// ModelSelector is mounted and shows the active model.
		await expect(page.getByTestId('model-selector')).toBeVisible();
		await expect(page.getByTestId('model-selector-current')).toContainText(/gpt2/i);
		await expect(page.getByTestId('model-selector-arch')).toHaveText('gpt2');

		// Block strip rendered.
		await expect(page.getByTestId('block-strip')).toBeVisible();

		// Embedding step exposes the positional kind on its always-rendered wrapper.
		await expect(page.getByTestId('embedding-step')).toHaveAttribute(
			'data-positional-kind',
			'absolute'
		);
		await expect(page.getByTestId('embedding-step')).toHaveAttribute('data-arch-kind', 'gpt2');

		// Top-k probability rail rendered.
		await expect(page.getByTestId('next-token-topk')).toBeVisible();
	});

	test('the app surfaces models from the workbench /models endpoint', async ({
		explainerPage: page
	}) => {
		const modelsRequest = page.waitForResponse(
			(resp) => resp.url().includes('/models') && resp.status() === 200,
			{ timeout: REAL_NDIF_TIMEOUT_MS }
		);
		await page.goto('/');
		const resp = await modelsRequest;
		expect(resp.status()).toBe(200);
		const body = await resp.json();
		expect(Array.isArray(body)).toBe(true);
		expect(body.length).toBeGreaterThan(0);
		const hasArchKind = body.every((m: any) => typeof m.arch_kind === 'string');
		expect(hasArchKind).toBe(true);
		const hasArchFields = body.every(
			(m: any) =>
				typeof m.n_heads === 'number' &&
				typeof m.d_model === 'number' &&
				typeof m.positional_kind === 'string'
		);
		expect(hasArchFields).toBe(true);
	});
});
