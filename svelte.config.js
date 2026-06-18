// import adapter from '@sveltejs/adapter-auto';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://kit.svelte.dev/docs/integrations#preprocessors
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		// adapter-auto only supports some environments, see https://kit.svelte.dev/docs/adapter-auto for a list.
		// If your environment is not supported, or you settled on a specific environment, switch out the adapter.
		// See https://kit.svelte.dev/docs/adapters for more information about adapters.
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: null,
			precompress: false,
			strict: false // Ignore errors about dynamic routes
		}),
		prerender: {
			// List the specific routes to prerender
			entries: ['/' /* other routes if needed */]
		},
		alias: {
			'~': './src'
		},
		paths: {
			// BASE_PATH is the source of truth for the deploy prefix; set
			// explicitly by the workbench Dockerfile when bundling the SPA
			// under /transformer-explainer/. Falls back to the upstream
			// gh-pages default for standalone production deploys, and to ''
			// when neither is set (local dev).
			base:
				process.env.BASE_PATH !== undefined
					? process.env.BASE_PATH
					: process.env.NODE_ENV === 'production'
						? '/transformer-explainer'
						: '',
			// Force absolute URLs (with base prefix) in the prerendered HTML.
			// Relative URLs require a trailing slash on the document URL, and
			// when this SPA is served behind a host that normalizes trailing
			// slashes (e.g. Next.js with `trailingSlash: false`), the redirect
			// dance produces a loop. Absolute URLs sidestep this entirely.
			relative: false
		}
	}
};

export default config;
