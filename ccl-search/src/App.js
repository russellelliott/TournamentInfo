import CCLSearch from './CCLSearch.js';

function App() {
  return (
    <div style={{ padding: '2rem', maxWidth: '900px', margin: '0 auto' }}>
      <h1>🧩 CCL Chess League Tracker</h1>
      <p>Find registration dates, standings, and results from chess.com</p>
      <CCLSearch />
    </div>
  );
}

export default App;
