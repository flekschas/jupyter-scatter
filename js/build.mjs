import esbuild from 'esbuild';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const tailwindPlugin = {
  name: 'tailwind',
  setup(build) {
    // Resolve the not-yet-existing built CSS to an absolute path so esbuild
    // doesn't fail when the file is missing on disk.
    build.onResolve({ filter: /styles\.built\.css$/ }, () => ({
      path: resolve(__dirname, 'src/styles.built.css'),
      namespace: 'file',
    }));

    // Intercept the built CSS file and rebuild Tailwind on every load.
    // This ensures esbuild always gets fresh CSS content, even in watch mode.
    build.onLoad({ filter: /styles\.built\.css$/ }, () => {
      execFileSync('npx', [
        'tailwindcss',
        '-i', 'src/styles.css',
        '-o', 'src/styles.built.css',
        '--minify',
      ], { cwd: __dirname, stdio: 'inherit' });
      return {
        contents: readFileSync(resolve(__dirname, 'src/styles.built.css'), 'utf8'),
        loader: 'css',
      };
    });
  },
};

const options = {
  entryPoints: ['src/index.js'],
  bundle: true,
  format: 'esm',
  jsx: 'automatic',
  loader: { '.jsx': 'jsx', '.tsx': 'tsx', '.ts': 'ts' },
  outfile: '../jscatter/bundle.js',
  plugins: [tailwindPlugin],
};

const watch = process.argv.includes('--watch');

if (watch) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
  console.log('Watching for changes...');
} else {
  await esbuild.build(options);
}
