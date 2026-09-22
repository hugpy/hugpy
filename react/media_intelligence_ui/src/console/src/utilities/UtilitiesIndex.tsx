// src/utilities/UtilitiesIndex.tsx
import { Link } from "react-router-dom";
import { pagesByCategory } from "./../imports/pages/pagesRegistry";
import "./../imports/pages/pagesBuiltin"; // side-effect: register everything

export default function UtilitiesIndex() {
  const grouped = pagesByCategory();
  return (
    <main>
      <h1>Utilities</h1>
      {Object.entries(grouped).map(([cat, pages]) => (
        <section key={cat}>
          <h2>{cat}</h2>
          <ul>
            {pages.map(p => (
              <li key={p.key}>
                <Link to={`/utilities/${p.key}`}>{p.title}</Link>
                {p.description && ` — ${p.description}`}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </main>
  );
}
