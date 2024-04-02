import { onMounted, onUnmounted } from 'vue';

export const videoColorModeSrcSwitcher = () => {
  let classObserver;

  onMounted(() => {
    const switchSrc = () => {
      const suffix = document.documentElement.classList.contains('dark')
        ? 'dark'
        : 'light';

      const videos = document.body.querySelectorAll('video');

      for (const video of videos) {
        const name = video.dataset.name;

        video
          .querySelector('source')
          ?.setAttribute('src', `/videos/${name}-${suffix}.mp4`);

        if (video.getAttribute('poster')) {
          video.setAttribute('poster', `/images/${name}-${suffix}.jpg`);
        }

        video.pause();
        video.currentTime = 0;
        video.load();
      }
    }

    classObserver = new window.MutationObserver((mutations) => {
      mutations.forEach((mu) => {
        if (mu.type !== 'attributes' && mu.attributeName !== 'class') return;
        switchSrc();
      });
    });
    classObserver.observe(document.documentElement, {attributes: true});
    switchSrc();
  });

  onUnmounted(() => {
    if (classObserver) classObserver.disconnect();
  });
}

export const videoPlayOnHover = () => {
  function mouseEnterHandler() { this.play(); }
  function mouseLeaveHandler() { this.pause(); }

  onMounted(() => {
    for (const video of document.body.querySelectorAll('video')) {
      video.addEventListener('mouseenter', mouseEnterHandler);
      video.addEventListener('mouseleave', mouseLeaveHandler);
    }
  });

  onUnmounted(() => {
    for (const video of document.body.querySelectorAll('video')) {
      video.removeEventListener('mouseenter', mouseEnterHandler);
      video.removeEventListener('mouseleave', mouseLeaveHandler);
    }
  });
}
