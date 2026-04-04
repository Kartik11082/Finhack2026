import { createContext, useContext, useState, useEffect } from "react";

const DataContext = createContext(null);

/** Parse a CSV string into an array of objects using the header row as keys. */
function parseCsv(text) {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(",").map((v) => v.trim());
    const obj = {};
    headers.forEach((h, i) => {
      const raw = values[i] ?? "";
      const num = Number(raw);
      obj[h] = raw !== "" && !Number.isNaN(num) ? num : raw;
    });
    return obj;
  });
}

export function DataProvider({ children }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadAll() {
      try {
        const [
          comparisonRes,
          fiBaseRes,
          fiEnhRes,
          foldRes,
          enhancedCsvRes,
        ] = await Promise.all([
          fetch("/data/model_comparison.json"),
          fetch("/data/feature_importance_baseline.json"),
          fetch("/data/feature_importance_enhanced.json"),
          fetch("/data/fold_results.json"),
          fetch("/data/dataset_enhanced.csv"),
        ]);

        // Check each response individually for clear error messages
        if (!comparisonRes.ok)
          throw new Error(`Failed to load model_comparison.json: ${comparisonRes.status}`);
        if (!fiBaseRes.ok)
          throw new Error(`Failed to load feature_importance_baseline.json: ${fiBaseRes.status}`);
        if (!fiEnhRes.ok)
          throw new Error(`Failed to load feature_importance_enhanced.json: ${fiEnhRes.status}`);
        if (!foldRes.ok)
          throw new Error(`Failed to load fold_results.json: ${foldRes.status}`);

        const comparison = await comparisonRes.json();
        const fiBaseline = await fiBaseRes.json();
        const fiEnhanced = await fiEnhRes.json();
        const foldResults = await foldRes.json();

        // Enhanced CSV is optional — sentiment may not exist yet
        let enhancedRows = [];
        if (enhancedCsvRes.ok) {
          const csvText = await enhancedCsvRes.text();
          if (csvText.trim().length > 0) {
            enhancedRows = parseCsv(csvText);
          }
        }

        setData({
          comparison,
          fiBaseline,
          fiEnhanced,
          foldResults,
          enhancedRows,
        });
      } catch (err) {
        setError(err.message);
      }
    }

    loadAll();
  }, []);

  if (error) {
    return <div className="error-msg">Data load error: {error}</div>;
  }

  if (!data) {
    return <div className="loading-msg">Loading dashboard data…</div>;
  }

  return <DataContext.Provider value={data}>{children}</DataContext.Provider>;
}

export function useData() {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error("useData must be used inside DataProvider");
  return ctx;
}
