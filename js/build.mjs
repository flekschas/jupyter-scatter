import esbuild from 'esbuild';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

const tailwindPlugin = {
  name: 'tailwind',
  setup(build) {
    // Intercept the built CSS file and rebuild Tailwind on every load.
    // This ensures esbuild always gets fresh CSS content, even in watch mode.
    build.onLoad({ filter: /styles\.built\.css$/ }, (args) => {
      execFileSync('npx', [
        'tailwindcss',
        '-i', 'src/styles.css',
        '-o', 'src/styles.built.css',
        '--minify',
      ], { stdio: 'inherit' });
      return {
        contents: readFileSync(args.path, 'utf8'),
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
