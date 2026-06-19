import { test, expect, REAL_NDIF_TIMEOUT_MS, waitForBootstrap } from './fixtures';

/**
 * Generic "page-load happy path" smoke. The runtime probe in +page.svelte picks
 * whichever model from /models actually completes a forward pass, so this test
 * stays agnostic about which arch ends up active — it just asserts that the
 * page successfully picked one, rendered it, and exposed consistent arch
 * metadata.
 *
 * Originally this hardcoded openai-community/gpt2 as the default, but gpt-2
 * isn't always HOT on NDIF (the key's pinned model set drifts over time).
 */
test.describe('forward_pass page-load happy path', () => {
	test('runtime probe lands on a working model and the page renders it', async ({
		explainerPage: page
	}) => {
		// The page-load probe retries each candidate a few times on transient
		// NDIF errors, so the bootstrap can take longer than a single
		// REAL_NDIF_TIMEOUT_MS budget when NDIF is degraded. Allow 5×.
		test.setTimeout(REAL_NDIF_TIMEOUT_MS * 5);

		await page.goto('/');
		await waitForBootstrap(page);

		// The probe loop fires /forward_pass/start for each candidate until one
		// succeeds. Wait for at least one start request to land — that's the
		// signal that bootstrap is past the /models fetch.
		const forwardPassRequest = page.waitForResponse(
			(resp) =>
				resp.url().includes('/forward_pass/') &&
				(resp.url().includes('/start') || resp.url().includes('/results/')) &&
				resp.status() < 500,
			{ timeout: REAL_NDIF_TIMEOUT_MS }
		);
		const resp = await forwardPassRequest;
		expect(resp.status()).toBeLessThan(500);

		// Wait until the selector shows a non-placeholder model name. The
		// probe loop may try several candidates before landing on one — we
		// allow the full NDIF timeout for that loop to complete.
		const selectorCurrent = page.getByTestId('model-selector-current');
		await expect(selectorCurrent).toBeVisible();
		await expect(selectorCurrent).not.toHaveText(/^—?$/, { timeout: REAL_NDIF_TIMEOUT_MS });

		// Whatever was picked, the embedding step's data-arch-kind and
		// data-positional-kind must be one of the known supported pairs.
		const archBadge = page.getByTestId('model-selector-arch');
		const archKind = (await archBadge.textContent())?.trim();
		expect(archKind, 'expected a known arch kind').toMatch(/^(gpt2|gptj|llama)$/);

		const embedding = page.getByTestId('embedding-step');
		await expect(embedding).toHaveAttribute('data-arch-kind', archKind!);
		const expectedPositional = archKind === 'gpt2' ? 'absolute' : 'rope';
		await expect(embedding).toHaveAttribute('data-positional-kind', expectedPositional);

		// Block strip rendered.
		await expect(page.getByTestId('block-strip')).toBeVisible();

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
