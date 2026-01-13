document.addEventListener('DOMContentLoaded', () => {

  // DOWNLOAD BUTTON LOADING POPUP
  const downloadLink = document.getElementById('download-link');
  const loadingPopup = document.getElementById('loading-popup');

  downloadLink.addEventListener('click', function (e) {
    loadingPopup.style.display = 'flex';

    // Hide popup after 3 seconds (simulate download start)
    setTimeout(() => {
      loadingPopup.style.display = 'none';
    }, 3000);
  });
});

