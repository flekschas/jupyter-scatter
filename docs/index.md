---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

title: Jupyter Scatter
titleTemplate: An Interactive Scatter Plot Widget

hero:
  name: "Jupyter Scatter"
  text: "An Interactive Scatter Plot Widget"
  tagline: "Explore datasets with millions of data points with ease in Jupyter Notebook, Lab, and Google Colab."
  image:
    src: /teaser.jpg
    alt: Jupyter Scatter
  actions:
    - theme: brand
      text: Get Started
      link: /get-started
    # - theme: alt
    #   text: Examples
    #   link: /api
    - theme: alt
      text: API
      link: /api

features:
  - title: Interactive
    details: Pan, zoom, and select data points interactively with your mouse or through the Python API.
    icon:
      dark: /images/icon-feature-interactive-dark.svg
      light: /images/icon-feature-interactive-light.svg
    
  - title: Scalable
    details: Plot up to several millions data points smoothly thanks to WebGL rendering.
    icon:
      dark: /images/icon-feature-scalable-dark.svg
      light: /images/icon-feature-scalable-light.svg
    
  - title: Interlinked
    details: Synchronize the view, hover, and selection across multiple scatter plot instances.
    icon:
      dark: /images/icon-feature-interlinked-dark.svg
      light: /images/icon-feature-interlinked-light.svg
    
  - title: Effective Defaults
    details: Rely on Jupyter Scatter to choose perceptually effective point colors and opacity by default.
    
  - title: Friendly API
    details: Enjoy a readable API that integrates deeply with Pandas DataFrames.
    
  - title: Integratable
    details: Use Jupyter Scatter in your own widgets by observing its traitlets.
---
