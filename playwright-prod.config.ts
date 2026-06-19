import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
	testDir: './tests',
	fullyParallel: false,
	forbidOnly: !!process.env.CI,
	retries: 0,
	workers: 1,
	reporter: process.env.CI ? 'dot' : 'list',
	use: {
		// vite preview serves the production build under the configured base path,
		// so we hit /transformer-explainer/ instead of /
		baseURL: 'http://localhost:4173/transformer-explainer/',
		trace: 'on-first-retry',
		screenshot: 'only-on-failure',
		ignoreHTTPSErrors: true
	},
	projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
	webServer: {
		// `vite preview` serves what `npm run build` produced; if we want to test
		// the real deploy shape, this is the closest local approximation.
		command: 'BASE_PATH=/transformer-explainer npm run preview -- --host 127.0.0.1 --port 4173',
		url: 'http://localhost:4173/transformer-explainer/',
		reuseExistingServer: !process.env.CI,
		timeout: 120_000
	}
});
