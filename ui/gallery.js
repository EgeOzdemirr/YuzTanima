const grid = document.getElementById("gallery-grid");
const plateList = document.getElementById("plate-list");

function renderPersons(persons) {
  grid.innerHTML = "";
  if (!persons || persons.length === 0) {
    const ph = document.createElement("div");
    ph.className = "placeholder";
    ph.style.position = "static";
    ph.style.minHeight = "120px";
    ph.textContent = "Kayıt yok";
    grid.appendChild(ph);
    return;
  }

  const fixedPlates = ["34 AB 123", "06 CD 456"];
  persons.forEach((p) => {
    const card = document.createElement("div");
    card.className = "thumb-card";

    const img = document.createElement("img");
    img.src = p.photoPath;
    img.alt = p.personId;
    card.appendChild(img);

    const name = document.createElement("div");
    name.className = "thumb-name";
    name.textContent = p.personId;
    card.appendChild(name);

    grid.appendChild(card);
  });
  renderPlates(fixedPlates);
}

async function loadGallery() {
  try {
    const res = await fetch("/data/gallery/gallery.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderPersons(data.persons || []);
  } catch (err) {
    console.error("gallery load error", err);
    grid.innerHTML = '<div class="placeholder" style="position:static;min-height:120px">Galeri yüklenemedi</div>';
  }
}

loadGallery();

function renderPlates(plates) {
  if (!plateList) return;
  plateList.innerHTML = "";
  const base = plates && plates.length ? plates : ["34 AB 123", "06 CD 456"];
  base.forEach((p) => {
    const pill = document.createElement("div");
    pill.className = "thumb-plate plate-pill";
    pill.textContent = p;
    plateList.appendChild(pill);
  });
}
