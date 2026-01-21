document.addEventListener('DOMContentLoaded', function () {
  // Filtering
  const filterBtns = document.querySelectorAll('.gallery-filters .filter-btn');
  const items = document.querySelectorAll('.gallery-item');

  filterBtns.forEach(b => b.addEventListener('click', () => {
    filterBtns.forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    const f = b.getAttribute('data-filter');
    items.forEach(it => {
      const tags = it.getAttribute('data-tags').split(' ');
      if (f === 'all' || tags.includes(f)) it.style.display = '';
      else it.style.display = 'none';
    });
  }));

  // Modal
  const modal = document.createElement('div');
  modal.className = 'gallery-modal';
  modal.innerHTML = '<div class="content"><button class="close-btn">Close</button><div class="hero"><img src="" alt=""><div class="meta"><h3 class="title"></h3><p class="desc"></p><p class="links"></p></div></div></div>';
  document.body.appendChild(modal);

  function openModal(imgSrc, title, desc, linksHtml) {
    modal.style.display = 'flex';
    modal.querySelector('img').src = imgSrc;
    modal.querySelector('.title').textContent = title;
    modal.querySelector('.desc').textContent = desc || '';
    modal.querySelector('.links').innerHTML = linksHtml || '';
  }

  function closeModal() { modal.style.display = 'none'; }

  modal.querySelector('.close-btn').addEventListener('click', closeModal);
  modal.addEventListener('click', function (e) { if (e.target === modal) closeModal(); });

  document.querySelectorAll('.gallery-item .view-btn').forEach(btn => {
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      const parent = btn.closest('.gallery-item');
      const img = parent.querySelector('img').src;
      const title = parent.getAttribute('data-title');
      const desc = parent.getAttribute('data-desc');
      const links = '<a href="' + (btn.getAttribute('data-href')||'#') + '">Open page</a>';
      openModal(img, title, desc, links);
    });
  });

});
