(function () {
  document.addEventListener('DOMContentLoaded', function () {
    var repoUrl = 'https://github.com/ThienNguyen3001/scikit-clarans';
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" aria-hidden="true">'
      + '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.54.1.73-.23.73-.5 0-.24-.01-.87-.01-1.7-2.22.48-2.69-1.07-2.69-1.07-.45-1.17-1.11-1.48-1.11-1.48-.91-.62.07-.61.07-.61 1.01.07 1.54 1.04 1.54 1.04.89 1.52 2.34 1.08 2.91.83.09-.65.35-1.08.64-1.33-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.36.31.68.92.68 1.85 0 1.33-.01 2.41-.01 2.74 0 .27.19.59.74.49C13.71 14.53 16 11.53 16 8c0-4.42-3.58-8-8-8z"/></svg>';

    function insertGitHubLink(targetElement) {
      if (!targetElement) return false;
      var a = document.createElement('a');
      a.href = repoUrl;
      a.className = 'github-link';
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.title = 'View project on GitHub';
      a.setAttribute('aria-label', 'View project on GitHub');
      a.innerHTML = svg;
      targetElement.appendChild(a);
      return true;
    }

    // Try to find the 'View page source' link and insert next to it
    var anchors = Array.from(document.querySelectorAll('a'));
    var viewSourceAnchor = anchors.find(function (x) {
      var t = (x.textContent || '').trim();
      return t === 'View page source' || t === 'View page source.';
    });

    if (viewSourceAnchor) {
      var parent = viewSourceAnchor.parentElement;
      if (parent) {
        insertGitHubLink(parent);
        return;
      }
    }

    // Fallback: try to insert into the top navigation bar
    var topNav = document.querySelector('.wy-side-nav-search') || document.querySelector('.wy-nav-top');
    if (topNav) {
      insertGitHubLink(topNav);
      return;
    }

    // Last fallback: append to document header
    var header = document.querySelector('header') || document.body;
    insertGitHubLink(header);
  });
})();
