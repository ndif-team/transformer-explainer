import { test as base, expect, type Page } from '@playwright/test';

/**
 * Fixtures for E2E tests that hit the real workbench API and the real NDIF
 * service. No mocks — we want the workflow exercised end to end against a
 * live model deployment.
 *
 * Tests assume the workbench FastAPI backend is running at the URL given by
 * VITE_WORKBENCH_API (default http://localhost:8000) with REMOTE=true and
 * NDIF_API_KEY set. The Svelte dev server is booted by playwright via
 * webServer.
 */
export const REAL_NDIF_TIMEOUT_MS = 120_000;

const DEBUG_NETWORK = !!process.env.DEBUG_TESTS;

export const test = base.extend<{ explainerPage: Page }>({
	explainerPage: async ({ page }, runFixture) => {
		page.on('pageerror', (err) => {
			console.error('[browser pageerror]', err.message);
		});
		page.on('console', (msg) => {
			if (msg.type() === 'error' || msg.type() === 'warning') {
				console.error(`[browser ${msg.type()}]`, msg.text());
			}
		});
		if (DEBUG_NETWORK) {
			page.on('request', (req) => {
				const url = req.url();
				if (url.includes('localhost:8000') || url.includes('ndif')) {
					console.log(`→ ${req.method()} ${url}`);
				}
			});
			page.on('response', async (resp) => {
				const url = resp.url();
				if (url.includes('localhost:8000') || url.includes('ndif')) {
					console.log(`← ${resp.status()} ${url}`);
				}
			});
		}
		await runFixture(page);
	}
});

export { expect };

/**
 * Wait for the app's initial bootstrap: /models to have resolved and the page
 * to have rendered the first attention matrix from cached or live data.
 */
export async function waitForBootstrap(page: Page) {
	// The legacy app paints cached ex0 data on first render and then re-renders
	// with the real API response. Either way, the first block's attention
	// matrix should be present once SSR + hydration are done.
	await expect(page).toHaveTitle(/Transformer Explainer/i, { timeout: 30_000 });
}
