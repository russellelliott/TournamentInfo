import { useState } from 'react';
import { load } from 'cheerio';

export default function CCLSearch() {
  const [season, setSeason] = useState('spring');
  const [year, setYear] = useState('2026');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [scrapedData, setScrapedData] = useState([]);
  const [loading, setLoading] = useState(false);

  const search = async () => {
    const query = `Collegiate Chess League ${season.charAt(0).toUpperCase() + season.slice(1)} ${year} site:chess.com`;
    setLoading(true);
    setScrapedData([]);
    
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
      
      const rawSources = data.citations || [];
      
      // Filter sources: must contain year and season, must not contain 'india'
      const filteredSources = rawSources.filter(url => {
        const lowerUrl = url.toLowerCase();
        const lowerSeason = season.toLowerCase();
        return lowerUrl.includes(year) && lowerUrl.includes(lowerSeason) && !lowerUrl.includes('india');
      });
      
      setSources(filteredSources);

      // Scrape filtered sources
      const results = [];
      for (const url of filteredSources) {
        try {
          // Use a CORS proxy to bypass CORS restrictions
          const proxyUrl = `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`;
          const res = await fetch(proxyUrl);
          if (!res.ok) throw new Error(`Failed to fetch via proxy: ${res.status}`);
          
          const html = await res.text();
          const $ = load(html);
          
          const result = {
            url,
            pdf: [],
            instructions: [],
            registration: [],
            fairPlay: [],
            platform: []
          };

          $('a').each((i, el) => {
            const href = $(el).attr('href');
            const text = $(el).text().toLowerCase();
            if (!href) return;

            // Filter out generic chess.com registration/login links
            if (href.includes('chess.com/register') || href.includes('chess.com/login')) return;

            const addLink = (arr, link) => {
                if (!arr.includes(link)) arr.push(link);
            };

            if (text.includes('pdf') || href.toLowerCase().includes('.pdf') || href.includes('drive.google.com') || (href.includes('docs.google.com') && !href.includes('/forms/'))) addLink(result.pdf, href);
            if (text.includes('instruction')) addLink(result.instructions, href);
            if (text.includes('registration') || text.includes('register') || text.includes('sign up') || href.includes('forms.gle') || href.includes('docs.google.com/forms')) addLink(result.registration, href);
            if (text.includes('fair play')) addLink(result.fairPlay, href);
            if (text.includes('platform') || text.includes('chess.com') || text.includes('tornelo')) addLink(result.platform, href);
          });
          
          results.push(result);
        } catch (e) {
          console.error(`Failed to scrape ${url}`, e);
          results.push({ url, error: 'Failed to scrape (likely CORS blocked)' });
        }
      }
      setScrapedData(results);

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

          {scrapedData.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <h4 style={{ margin: '0 0 0.5rem 0' }}>Scraped Data</h4>
              {scrapedData.map((item, i) => (
                <div key={i} style={{ marginBottom: '1rem', padding: '0.5rem', border: '1px solid #eee', borderRadius: '4px' }}>
                  <div style={{ marginBottom: '0.5rem', wordBreak: 'break-all' }}>
                    <strong>Source: </strong>
                    <a href={item.url} target="_blank" rel="noopener noreferrer">{item.url}</a>
                  </div>
                  {item.error ? (
                    <div style={{ color: 'red' }}>{item.error}</div>
                  ) : (
                    <ul style={{ fontSize: '0.9rem', paddingLeft: '1.5rem', margin: 0 }}>
                      {item.pdf.length > 0 && <li><strong>PDFs:</strong> {item.pdf.map((l, idx) => <a key={idx} href={l} style={{marginRight: '5px'}}>{l}</a>)}</li>}
                      {item.instructions.length > 0 && <li><strong>Instructions:</strong> {item.instructions.map((l, idx) => <a key={idx} href={l} style={{marginRight: '5px'}}>{l}</a>)}</li>}
                      {item.registration.length > 0 && <li><strong>Registration:</strong> {item.registration.map((l, idx) => <a key={idx} href={l} style={{marginRight: '5px'}}>{l}</a>)}</li>}
                      {item.fairPlay.length > 0 && <li><strong>Fair Play:</strong> {item.fairPlay.map((l, idx) => <a key={idx} href={l} style={{marginRight: '5px'}}>{l}</a>)}</li>}
                      {item.platform.length > 0 && <li><strong>Platform:</strong> {item.platform.map((l, idx) => <a key={idx} href={l} style={{marginRight: '5px'}}>{l}</a>)}</li>}
                      {item.pdf.length === 0 && item.instructions.length === 0 && item.registration.length === 0 && item.fairPlay.length === 0 && item.platform.length === 0 && (
                        <li>No specific links found.</li>
                      )}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
