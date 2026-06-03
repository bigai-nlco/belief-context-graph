import "./style.css";

const app = document.querySelector<HTMLDivElement>("#app");

if (app) {
  app.innerHTML = `
    <main class="shell">
      <section>
        <p class="eyebrow">Belief Context Graph</p>
        <h1>Dashboard scaffold</h1>
        <p class="summary">Vite is initialized. Dashboard implementation is intentionally empty.</p>
      </section>
    </main>
  `;
}
