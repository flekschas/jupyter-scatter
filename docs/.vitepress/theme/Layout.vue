<script setup>
  import DefaultTheme from "vitepress/theme";

  import { videoColorModeSrcSwitcher } from '../../utils';

  videoColorModeSrcSwitcher();
</script>

<template>
  <DefaultTheme.Layout>
    <template #home-hero-image>
      <video
        autoplay
        loop
        muted
        playsinline
        data-name="teaser"
        poster="/images/teaser-dark.jpg"
      >
        <source
          src="/videos/teaser-dark.mp4"
          type="video/mp4"
        />
      </video>
    </template>
    <template #home-features-after>
      <section id="scipy-talk">
        <div class="container">
          <h3>Learn About Jupyter Scatter</h3>
          <div class="card">
            <div class="youtube-container">
              <span />
              <iframe
                src="https://www.youtube.com/embed/RyC5ixtQG-Q?si=Vn1EDWvx1LHGyys7"
                title="YouTube Player Showing Jupyter Scatter talk from the SciPy 2023 conference"
                frameborder="0"
                scrolling="no"
                allow="fullscreen"
                allowfullscreen
              />
            </div>
          </div>
          <p>In this talk you'll learn how to use Jupyter Scatter for exploring large-scale datasets using geospatial, single-cell, image, and text embeddings.</p>
        </div>
      </section>
    </template>
  </DefaultTheme.Layout>
</template>

<style scoped>
  video {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    border-radius: 0.5rem;
    padding: 0.25rem 0.5rem;
    box-shadow: 0 0.05rem 0.25rem 0 rgba(0, 0, 0, 0.1), 0 0.2rem 1rem 0 rgba(0, 0, 0, 0.1);
    background: #FDFDFD;
  }

  :root.dark video {
    box-shadow: 0 0.05rem 0.25rem 0 rgba(0, 0, 0, 0.5), 0 0.2rem 1rem 0 rgba(0, 0, 0, 0.33);
    background: #1B1B1E;
  }

  #scipy-talk {
    position: relative;
    padding: 48px 24px;
  }

  #scipy-talk .container {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    max-width: 1152px;
    margin: 0 auto;
  }

  #scipy-talk h3 {
    font-size: 2rem;
    line-height: 3rem;
    font-weight: 700;
    color: var(--vp-c-text-1);
  }

  #scipy-talk p {
    font-size: 1rem;
    color: var(--vp-c-text-2);
    width: 100%;
    text-align: center;
  }

  :root.dark #scipy-talk p {
    color: var(--vp-c-text-3);
  }

  #scipy-talk .card {
    position: relative;
    margin: 1rem 0;
    padding: 12px;
    width: 100%;
    border: 1px solid var(--vp-c-bg-soft);
    border-radius: 12px;
    background-color: var(--vp-c-bg-soft);
  }

  #scipy-talk .youtube-container {
    position: relative;
  }

  #scipy-talk .youtube-container span {
    display: block;
    padding-top: 60%;
  }

  #scipy-talk .youtube-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }

  @media (min-width: 640px) {
    #scipy-talk {
      padding: 64px 48px;
    }

    #scipy-talk .card, #scipy-talk p {
      width: 75%;
    }

    .video {
      max-height: unset;
    }
  }

  @media (min-width: 960px) {
    #scipy-talk {
      padding: 96px 64px;
    }

    #scipy-talk .card, #scipy-talk p {
      width: 66%;
    }
  }
</style>
