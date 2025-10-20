const API_BASE = "https://rag-1-05un.onrender.com/";

async function uploadDocument() {
    const content = document.getElementById("docContent").value;
    const source = document.getElementById("docSource").value;
    const title = document.getElementById("docTitle").value;
    
    if (!content || !source || !title) {
        showStatus("Please fill in all fields", "error");
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content, source, title })
        });
        
        if (response.ok) {
            const data = await response.json();
            showStatus(`Success! Created ${data.chunks_created} chunks`, "success");
            document.getElementById("docContent").value = "";
            document.getElementById("docSource").value = "";
            document.getElementById("docTitle").value = "";
        } else {
            showStatus("Upload failed", "error");
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, "error");
    }
}

async function submitQuery() {
    const query = document.getElementById("queryInput").value;
    if (!query.trim()) return;
    
    document.getElementById("loading").style.display = "block";
    document.getElementById("results").style.display = "none";
    
    try {
        const startTime = performance.now();
        const response = await fetch(`${API_BASE}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
        });
        
        if (response.ok) {
            const data = await response.json();
            displayResults(data);
        } else {
            alert("Query failed");
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        document.getElementById("loading").style.display = "none";
    }
}

function displayResults(data) {
    // Timing info
    const timing = document.getElementById("timing");
    timing.innerHTML = `⏱️ Execution time: ${data.execution_time.toFixed(2)}s | 📊 Tokens: ~${data.token_estimate}`;
    
    // Answer
    document.getElementById("answer").textContent = data.answer;
    
    // Citations - Only source and title
    const citationsDiv = document.getElementById("citations");
    citationsDiv.innerHTML = data.citations.map(c => `
        <div class="citation-item">
            <strong>[${c.index}] ${c.title}</strong>
            <small>${c.source}</small>
        </div>
    `).join("");
    
    // Sources
    const sourcesDiv = document.getElementById("sources");
    sourcesDiv.innerHTML = data.sources.map(s => `
        <div class="source-item">
            <strong>${s.metadata.title}</strong>
            <span class="source-score">Score: ${(s.score * 100).toFixed(1)}%</span>
            <small>Source: ${s.metadata.source}</small>
        </div>
    `).join("");
    
    document.getElementById("results").style.display = "block";
}

function showStatus(message, type) {
    const el = document.getElementById("uploadStatus");
    el.textContent = message;
    el.className = type;
}

document.getElementById("queryInput").addEventListener("keypress", e => {
    if (e.key === "Enter") submitQuery();
});