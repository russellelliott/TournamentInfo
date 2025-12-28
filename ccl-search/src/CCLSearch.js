import { useState } from 'react';

export default function CCLSearch() {
  const [season, setSeason] = useState('spring');
  const [year, setYear] = useState('2026');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  const search = async () => {
    const query = `Collegiate Chess League ${season.charAt(0).toUpperCase() + season.slice(1)} ${year} site:chess.com`;
    setLoading(true);
    
    try {
      const response = await fetch('https://api.perplexity.ai/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.REACT_APP_PERPLEXITY_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'sonar-pro',
          messages: [{ role: 'user', content: query }]
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      setAnswer(data.choices[0]?.message?.content || '');
      setSources(data.citations || []);
    } catch (error) {
      setAnswer('Error: Check API key');
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '1rem', border: '1px solid #ccc', borderRadius: '8px' }}>
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ marginRight: '0.5rem' }}>Season: 
          <select 
            value={season} 
            onChange={(e) => setSeason(e.target.value)}
            style={{ marginLeft: '0.5rem', padding: '0.25rem' }}
          >
            <option value="fall">Fall</option>
            <option value="spring">Spring</option>
            <option value="summer">Summer</option>
          </select>
        </label>
        <label style={{ marginLeft: '1rem' }}>Year: 
          <input 
            type="number" 
            value={year} 
            onChange={(e) => setYear(e.target.value)}
            min="2025" 
            max="2030"
            style={{ marginLeft: '0.5rem', width: '70px', padding: '0.25rem' }}
          />
        </label>
        <button 
          onClick={search} 
          disabled={loading}
          style={{ 
            marginLeft: '1rem', 
            padding: '0.5rem 1rem',
            background: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? 'Searching...' : 'Search CCL'}
        </button>
      </div>
      
      {answer && (
        <>
          <h3 style={{ margin: '0 0 0.5rem 0' }}>Results for: {season} {year}</h3>
          <p style={{ whiteSpace: 'pre-wrap', marginBottom: '1rem' }}>{answer}</p>
          
          {sources.length > 0 && (
            <div>
              <strong>Sources:</strong>
              <ul style={{ margin: '0.5rem 0', paddingLeft: '1.5rem' }}>
                {sources.map((source, i) => (
                  <li key={i}>
                    <a 
                      href={typeof source === 'string' ? source : source.url} 
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: '#007bff' }}
                    >
                      {source.title || source.url || source}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
}
