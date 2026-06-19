import { test, expect, REAL_NDIF_TIMEOUT_MS, waitForBootstrap } from './fixtures';

/**
 * Pure-UI test for the ModelSelector dropdown. Doesn't care how many models
 * are available — even with just GPT-2 in /models, clicking the button must
 * reveal the list and clicking outside must close it.
 */
// First test in a fresh dev-server run pays the vite first-compile cost
// (~30s). Lift the per-test timeout above the default 30s so we don't bail
// before reaching the assertion.
test.setTimeout(120_000);

test.describe('ModelSelector dropdown', () => {
	test('clicking the button reveals the list, click-outside closes it', async ({
		explainerPage: page
	}) => {
		await page.goto('/');
		await waitForBootstrap(page);

		// Wait for the initial forward_pass to settle so isModelRunning=false.
		await page.waitForResponse(
			(resp) => resp.url().includes('/forward_pass/') && resp.status() < 500,
			{ timeout: REAL_NDIF_TIMEOUT_MS }
		);

		const button = page.getByTestId('model-selector-button');
		await expect(button).toBeVisible();
		await expect(button).toBeEnabled();
		// Before any click the list shouldn't be in the DOM.
		await expect(page.getByTestId('model-selector-list')).toHaveCount(0);

		await button.click();
		await expect(page.getByTestId('model-selector-list')).toBeVisible({ timeout: 5000 });

		const options = page.locator('[data-testid^="model-selector-option-"]');
		await expect(options.first()).toBeVisible();
		expect(await options.count()).toBeGreaterThan(0);

		// Click outside closes the menu.
		await page.mouse.click(5, 200);
		await expect(page.getByTestId('model-selector-list')).toHaveCount(0, { timeout: 5000 });
	});

	test('dropdown sits above scene layers (z-index check)', async ({ explainerPage: page }) => {
		await page.goto('/');
		await waitForBootstrap(page);
		await page.waitForResponse(
			(resp) => resp.url().includes('/forward_pass/') && resp.status() < 500,
			{ timeout: REAL_NDIF_TIMEOUT_MS }
		);
		await page.getByTestId('model-selector-button').click();
		const list = page.getByTestId('model-selector-list');
		await expect(list).toBeVisible();
		// Computed z-index must be high enough to clear the scene (which uses
		// indices up to ~1000 in styles/variables.scss).
		const z = await list.evaluate((el) => parseInt(getComputedStyle(el).zIndex, 10));
		expect(Number.isFinite(z)).toBe(true);
		expect(z).toBeGreaterThan(1000);
	});
});
