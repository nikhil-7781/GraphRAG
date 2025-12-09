/**
 * GraphLLM Frontend JavaScript
 * Handles user interactions, API calls, and dynamic UI updates
 */

// ========== Global State ==========
let currentPdfId = null;
let graphData = { nodes: [], edges: [] };
let selectedNodeId = null;

// ========== API Configuration ==========
const API_BASE = window.location.origin;

// ========== Processing Overlay Functions ==========
function showProcessingOverlay(title = 'Processing PDF', message = 'Starting...', percent = 0) {
    const overlay = document.getElementById('processing-overlay');
    const titleEl = document.getElementById('processing-title');
    const messageEl = document.getElementById('processing-message');
    const percentEl = document.getElementById('processing-percent');
    const progressFill = document.getElementById('progress-fill');

    titleEl.textContent = title;
    messageEl.textContent = message;
    percentEl.textContent = `${percent}%`;
    progressFill.style.width = `${percent}%`;

    overlay.hidden = false;
}

function updateProcessingOverlay(message, percent) {
    const messageEl = document.getElementById('processing-message');
    const percentEl = document.getElementById('processing-percent');
    const progressFill = document.getElementById('progress-fill');

    messageEl.textContent = message;
    percentEl.textContent = `${percent}%`;
    progressFill.style.width = `${percent}%`;
}

function hideProcessingOverlay() {
    const overlay = document.getElementById('processing-overlay');
    overlay.hidden = true;
}

// ========== Utility Functions ==========
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        showNotification(error.message, 'error');
        throw error;
    }
}

function showNotification(message, type = 'info') {
    const statusEl = document.getElementById('upload-status');
    statusEl.textContent = message;
    statusEl.style.color = type === 'error' ? '#f44336' : type === 'success' ? '#4caf50' : '#4f9eff';

    setTimeout(() => {
        statusEl.textContent = '';
    }, 5000);
}

// ========== PDF Upload ==========
document.getElementById('pdf-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Show overlay immediately
    showProcessingOverlay('Uploading PDF', `Uploading ${file.name}...`, 0);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const result = await apiCall('/upload', {
            method: 'POST',
            body: formData
        });

        currentPdfId = result.pdf_id;
        updateProcessingOverlay('Upload complete, starting processing...', 5);

        // Poll for completion
        pollProcessingStatus(result.pdf_id);

    } catch (error) {
        hideProcessingOverlay();
        showNotification('Upload failed', 'error');
    }
});

async function pollProcessingStatus(pdfId) {
    const interval = setInterval(async () => {
        try {
            // Fetch detailed status for this PDF
            const status = await apiCall(`/status/${pdfId}`);

            // Update overlay with progress
            if (status.progress) {
                const { message, percent } = status.progress;
                updateProcessingOverlay(message, percent);
            }

            // Check if processing is complete
            if (status.status === 'completed') {
                clearInterval(interval);

                // Show completion message briefly
                updateProcessingOverlay(
                    `✓ Complete! ${status.num_nodes} nodes, ${status.num_edges} edges`,
                    100
                );

                // Load graph and hide overlay
                setTimeout(async () => {
                    hideProcessingOverlay();
                    await loadGraph();
                    await updateStats();
                    showNotification(`✓ Graph loaded: ${status.num_nodes} nodes, ${status.num_edges} edges`, 'success');
                }, 1500); // Show completion for 1.5s

            } else if (status.status === 'failed') {
                clearInterval(interval);
                hideProcessingOverlay();
                showNotification(`Error: ${status.error}`, 'error');
            }
        } catch (error) {
            clearInterval(interval);
            hideProcessingOverlay();
            showNotification('Failed to check status', 'error');
        }
    }, 1000); // Poll every 1 second for responsive updates

    // Stop polling after 5 minutes
    setTimeout(() => {
        clearInterval(interval);
        hideProcessingOverlay();
        showNotification('Processing timeout', 'error');
    }, 300000);
}

// ========== Graph Loading ==========
let network = null;

async function loadGraph() {
    try {
        const data = await apiCall('/graph');
        graphData = data;

        // Render interactive graph visualization
        renderGraph(data);

    } catch (error) {
        console.error('Failed to load graph:', error);
    }
}

function renderGraph(data) {
    const container = document.getElementById('graph-container');

    // Clear any existing content
    container.innerHTML = '';

    console.log(`Rendering graph: ${data.nodes.length} nodes, ${data.edges.length} edges`);

    // Get actual container dimensions
    const rect = container.getBoundingClientRect();
    const containerHeight = rect.height || 600; // Fallback to 600px
    const containerWidth = rect.width || 800;   // Fallback to 800px

    // Set explicit container styles to prevent overflow
    container.style.position = 'relative';
    container.style.width = containerWidth + 'px';
    container.style.height = containerHeight + 'px';
    container.style.overflow = 'hidden';

    // Prepare nodes for vis.js
    const visNodes = data.nodes.map(node => ({
        id: node.node_id,
        label: node.label,
        title: `${node.label}\nType: ${node.type}\nImportance: ${node.importance_score.toFixed(2)}`,
        value: node.importance_score * 20, // Size based on importance
        group: node.type,
        font: { color: '#e6eef8' }
    }));

    // Prepare edges for vis.js (thin, bright green, no arrows - undirected graph)
    const visEdges = data.edges.map(edge => ({
        from: edge.from || edge.from_node,  // Handle both alias and field name
        to: edge.to || edge.to_node,        // Handle both alias and field name
        label: edge.relation,
        title: `${edge.relation} (${edge.confidence.toFixed(2)})`,
        width: 1.5,  // Thin edges
        // No arrows for undirected graph
        color: {
            color: '#00ff00',  // BRIGHT NEON GREEN (most visible)
            highlight: '#ff00ff',  // Neon magenta when highlighted
            hover: '#ffff00',  // Yellow on hover
            opacity: 1.0  // Full opacity
        },
        font: {
            size: 12,
            color: '#ffffff',
            strokeWidth: 3,
            strokeColor: '#000000',
            background: 'rgba(0, 0, 0, 0.8)',
            bold: true
        }
    }));

    // Create vis.js network
    const graphData = {
        nodes: new vis.DataSet(visNodes),
        edges: new vis.DataSet(visEdges)
    };

    const options = {
        nodes: {
            shape: 'dot',
            scaling: {
                min: 10,
                max: 30
            },
            font: {
                size: 12,
                face: 'Arial',
                color: '#e6eef8'
            },
            borderWidth: 2,
            shadow: true
        },
        edges: {
            width: 1.5,  // Thin edges
            color: {
                color: '#00ff00',  // BRIGHT NEON GREEN (most visible against dark bg)
                highlight: '#ff00ff',  // Neon magenta when highlighted
                hover: '#ffff00',  // Yellow on hover
                opacity: 1.0  // Full opacity
            },
            arrows: {
                to: { enabled: false }  // No arrows - undirected graph
            },
            smooth: {
                type: 'continuous',
                roundness: 0.2  // Less curved = more visible
            },
            font: {
                size: 12,  // Moderate text size
                color: '#ffffff',  // White text
                strokeWidth: 3,  // Moderate outline
                strokeColor: '#000000',  // Black outline for readability
                align: 'top',  // Position above edge
                bold: true,
                background: 'rgba(0, 0, 0, 0.8)'  // Dark background for label
            },
            selectionWidth: 3,  // Moderately thicker when selected
            hoverWidth: 2.5,  // Slightly thicker on hover
            shadow: {
                enabled: true,
                color: 'rgba(0, 255, 0, 0.5)',  // Green glow
                size: 5,
                x: 0,
                y: 0
            }
        },
        groups: {
            concept: { color: { background: '#4f9eff', border: '#3d8ae6' } },
            function: { color: { background: '#9c27b0', border: '#7b1fa2' } },
            class: { color: { background: '#ff5722', border: '#e64a19' } },
            term: { color: { background: '#4caf50', border: '#388e3c' } },
            person: { color: { background: '#ff9800', border: '#f57c00' } },
            method: { color: { background: '#00bcd4', border: '#0097a7' } },
            entity: { color: { background: '#607d8b', border: '#455a64' } }
        },
        physics: {
            stabilization: { iterations: 200 },
            barnesHut: {
                gravitationalConstant: -8000,
                springConstant: 0.04,
                springLength: 95
            }
        },
        interaction: {
            hover: true,
            navigationButtons: true,
            keyboard: true
        },
        autoResize: false,  // Disable auto-resize to prevent infinite stretching
        height: containerHeight + 'px',
        width: containerWidth + 'px'
    };

    // Create network
    network = new vis.Network(container, graphData, options);

    // Prevent any further resize attempts
    if (network) {
        network.setOptions({ autoResize: false });
    }

    // Add click handler for nodes
    network.on('click', function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            selectNode(nodeId);
        }
    });
}

// ========== Node Selection ==========
window.selectNode = async function(nodeId) {
    selectedNodeId = nodeId;

    try {
        const nodeData = await apiCall(`/node/${nodeId}`);
        displayNodeDetails(nodeData);
    } catch (error) {
        console.error('Failed to load node details:', error);
    }
}

function displayNodeDetails(nodeData) {
    const content = document.getElementById('node-content');

    const sourcesHtml = nodeData.sources.map((source, i) => `
        <li>p.${source.page_number} - "${source.snippet}" <span style="color: #8b92a0;">(${source.chunk_id})</span></li>
    `).join('');

    const relatedHtml = nodeData.related_nodes.map(related => `
        <li onclick="selectNode('${related.node_id}')" style="cursor: pointer; padding: 0.5rem; background: #23262e; border-radius: 6px; margin-bottom: 0.25rem;">
            <strong>${related.label}</strong> - ${related.relation} (confidence: ${related.confidence.toFixed(2)})
        </li>
    `).join('');

    content.innerHTML = `
        <div class="node-info">
            <h3 class="node-label">${nodeData.label}</h3>
            <span class="badge">${nodeData.type}</span>

            <div class="node-summary">
                <h4>Summary</h4>
                <p>${nodeData.summary}</p>
            </div>

            <div class="node-sources">
                <h4>Sources</h4>
                <button class="expand-toggle" onclick="toggleSources()">Show Sources</button>
                <ul class="sources-list" id="sources-list" hidden>
                    ${sourcesHtml}
                </ul>
            </div>

            ${nodeData.related_nodes.length > 0 ? `
                <div class="related-nodes">
                    <h4>Related Nodes</h4>
                    <ul class="related-list">
                        ${relatedHtml}
                    </ul>
                </div>
            ` : ''}
        </div>
    `;
}

window.toggleSources = function() {
    const sourcesList = document.getElementById('sources-list');
    const toggle = document.querySelector('.expand-toggle');

    if (sourcesList.hidden) {
        sourcesList.hidden = false;
        toggle.textContent = 'Hide Sources';
    } else {
        sourcesList.hidden = true;
        toggle.textContent = 'Show Sources';
    }
}

document.getElementById('close-node-detail').addEventListener('click', () => {
    document.getElementById('node-content').innerHTML = '<p class="placeholder-text">Click a node in the graph to view details</p>';
    selectedNodeId = null;
});

// ========== Chat ==========
document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const query = input.value.trim();

    if (!query) return;
    if (!currentPdfId) {
        showNotification('Please upload a PDF first', 'error');
        return;
    }

    // Add user message to chat
    addMessageToChat('user', query);
    input.value = '';

    try {
        const includeCitations = document.getElementById('include-citations').checked;

        const response = await apiCall('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                pdf_id: currentPdfId,
                include_citations: includeCitations,
                max_sources: 5
            })
        });

        // Add assistant response
        addMessageToChat('assistant', response.answer, response.sources);

    } catch (error) {
        addMessageToChat('assistant', 'Sorry, I encountered an error processing your question.');
    }
}

function addMessageToChat(role, content, sources = []) {
    const messagesContainer = document.getElementById('chat-messages');

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    let html = `<p>${content}</p>`;

    if (sources && sources.length > 0) {
        html += '<div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.1);">';
        html += '<strong style="font-size: 0.875rem;">Sources:</strong><ul style="margin-top: 0.25rem; font-size: 0.875rem;">';
        sources.forEach(source => {
            html += `<li>p.${source.page_number}: "${source.snippet}"</li>`;
        });
        html += '</ul></div>';
    }

    messageDiv.innerHTML = html;
    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// ========== Stats Update ==========
async function updateStats() {
    try {
        const status = await apiCall('/admin/status');

        document.getElementById('stats-nodes').textContent = `Nodes: ${status.total_nodes}`;
        document.getElementById('stats-edges').textContent = `Edges: ${status.total_edges}`;
        document.getElementById('stats-chunks').textContent = `Chunks: ${status.total_chunks}`;
    } catch (error) {
        console.error('Failed to update stats:', error);
    }
}

// ========== Admin Controls ==========
document.getElementById('reindex-btn').addEventListener('click', async () => {
    if (!currentPdfId) {
        showNotification('No PDF to reindex', 'error');
        return;
    }

    if (!confirm('Reindex current PDF? This will take some time.')) return;

    try {
        // Show overlay for reindexing
        showProcessingOverlay('Reindexing PDF', 'Starting reindex...', 0);

        await apiCall(`/admin/reindex?pdf_id=${currentPdfId}`, { method: 'POST' });

        // Poll for completion
        pollProcessingStatus(currentPdfId);
    } catch (error) {
        hideProcessingOverlay();
        showNotification('Reindex failed', 'error');
    }
});

document.getElementById('clear-btn').addEventListener('click', async () => {
    if (!confirm('Clear all data? This cannot be undone!')) return;

    try {
        await apiCall('/admin/clear', { method: 'POST' });
        showNotification('All data cleared', 'success');

        // Reset UI
        currentPdfId = null;
        graphData = { nodes: [], edges: [] };
        document.getElementById('graph-container').innerHTML = '<div class="graph-placeholder"><p>Upload a PDF to generate a knowledge graph</p></div>';
        document.getElementById('node-content').innerHTML = '<p class="placeholder-text">Click a node in the graph to view details</p>';
        document.getElementById('chat-messages').innerHTML = '<div class="message system"><p>Ask questions about your uploaded PDF. Answers will cite page numbers.</p></div>';
        await updateStats();
    } catch (error) {
        showNotification('Clear failed', 'error');
    }
});

// ========== Graph Controls ==========
document.getElementById('zoom-in-btn').addEventListener('click', () => {
    if (network) {
        const scale = network.getScale();
        network.moveTo({ scale: scale * 1.2 });
    }
});

document.getElementById('zoom-out-btn').addEventListener('click', () => {
    if (network) {
        const scale = network.getScale();
        network.moveTo({ scale: scale * 0.8 });
    }
});

document.getElementById('reset-view-btn').addEventListener('click', () => {
    if (network) {
        network.fit();
    }
});

// ========== Initialization ==========
document.addEventListener('DOMContentLoaded', () => {
    updateStats();
    console.log('GraphLLM Frontend Initialized');
});
