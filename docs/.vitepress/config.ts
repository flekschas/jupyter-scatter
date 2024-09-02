import { defineConfig } from 'vitepress';

import pkg from '../../js/package.json';

// https://vitepress.dev/reference/site-config
export default defineConfig({
  lang: 'en-US',
  title: 'Jupyter Scatter',
  description: 'An interactive scatter plot widget for exploring datasets with millions of data points in Jupyter Notebook, Lab, and Google Colab.',
  lastUpdated: true,
  cleanUrls: true,
  head: [
    ['link', { rel: 'icon', href: '/favicon.svg' }],
    ['meta', { name: 'theme-color', content: '#FEC10E' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:locale', content: 'en' }],
    ['meta', { name: 'og:site_name', content: 'Jupyter Scatter' }],
    [
      'meta',
      { name: 'og:image', content: 'https://jupyter-scatter.dev/images/jupyter-scatter-og.jpg' }
    ],
    [
      'meta',
      {
        name: 'twitter:image',
        content: 'https://jupyter-scatter.dev/images/jupyter-scatter-twitter.jpg'
      }
    ],
    [
      'script',
      {
        'async': '',
        src: 'https://cdn.counter.dev/script.js',
        'data-id': 'c0cd84ef-1b57-4616-a6db-c26306f866b7',
        'data-utcoffset': '-4',
      }
    ],
  ],
  themeConfig: {
    logo: {
      light: '/images/logo-light.svg',
      dark: '/images/logo-dark.svg',
    },

    nav: [
      { text: 'Home', link: '/' },
      { text: 'Get Started', link: '/get-started' },
      // { text: 'Examples', link: '/examples' },
      { text: 'API', link: '/api' },
      {
        text: `v${pkg.version}`,
        items: [
          {
            text: 'pypi',
            link: 'https://pypi.org/project/jupyter-scatter/'
          },
          {
            text: 'Changelog',
            link: 'https://github.com/flekschas/jupyter-scatter/blob/main/CHANGELOG.md'
          }
        ]
      }
    ],

    sidebar: [
      {
        text: 'Get Started',
        items: [
          { text: 'Plotting', link: '/get-started' },
          { text: 'Interactions', link: '/interactions' },
          { text: 'Selections', link: '/selections' },
          { text: 'Link Multiple Scatter Plots', link: '/link-multiple-plots' },
          { text: 'Axes & Legends', link: '/axes-legends' },
          { text: 'Annotations', link: '/annotations' },
          { text: 'Tooltip', link: '/tooltip' },
          { text: 'Scales', link: '/scales' },
          { text: 'Connected Scatterplots', link: '/connected-scatterplots' },
          { text: 'Export as Image', link: '/export-image' },
        ]
      },
      // {
      //   text: 'Examples',
      //   items: [
      //     { text: 'Exploring LLM-based sentence embeddings', link: '/sentence-embeddings' },
      //     { text: 'Comparing multiple embedding methods of the Fashion MNIST dataset', link: '/image-embeddings' },
      //     { text: 'Browsing genomic data with HiGlass and loci embeddings', link: '/genomic-embeddings' },
      //     { text: 'Comparing a pair of single-cell embeddings by their label abundance differences', link: '/single-cell-embeddings' },
      //   ]
      // },
      {
        text: 'API Reference',
        link: '/api',
      },
      {
        text: 'Tutorial & Demos',
        link: 'https://github.com/flekschas/jupyter-scatter-tutorial',
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/flekschas/jupyter-scatter' }
    ],

    search: {
      provider: 'local'
    },

    footer: {
      message: 'Released under the <a href="https://github.com/flekschas/jupyter-scatter/blob/main/LICENSE">Apache License Version 2.0</a>.',
      copyright: 'Copyright © 2021-present <a href="https://lekschas.de" target="_blank" rel="noreferrer">Fritz Lekschas</a>.'
    },
  }
})
