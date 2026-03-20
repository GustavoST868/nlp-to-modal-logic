let lossChart = null;
let rougeChart = null;
let chatHistory = [];

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

function newChat() {
    chatHistory = [];
    document.getElementById('chat-history').innerHTML = '';
    document.getElementById('chat-history').classList.add('hidden');
    document.getElementById('welcome-view').classList.remove('hidden');
    document.getElementById('input-text').value = '';
    autoResize(document.getElementById('input-text'));
}

async function performTranslation() {
    const inputField = document.getElementById('input-text');
    const text = inputField.value.trim();
    
    if (!text) return;

    // UI State
    inputField.value = '';
    autoResize(inputField);
    document.getElementById('welcome-view').classList.add('hidden');
    document.getElementById('chat-history').classList.remove('hidden');

    // Add User Message
    addMessage(text, 'user');
    
    // Add Thinking Message
    const thinkingId = addMessage('...', 'assistant', true);

    try {
        const response = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const data = await response.json();
        
        // Remove Thinking
        removeMessage(thinkingId);

        if (response.ok) {
            addMessage(data.output, 'assistant');
            updateSidebar(text);
        } else {
            showToast(data.error || "Erro na tradução.", "error");
        }
    } catch (error) {
        removeMessage(thinkingId);
        showToast("Erro de conexão.", "error");
    }
}

function addMessage(content, role, isThinking = false) {
    const historyContainer = document.getElementById('chat-history');
    const id = 'msg-' + Date.now();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    messageDiv.id = id;
    
    const icon = role === 'user' ? 'ph ph-user' : 'ph ph-sparkle';
    
    messageDiv.innerHTML = `
        <div class="avatar"><i class="${icon}"></i></div>
        <div class="msg-content">${content}</div>
    `;
    
    historyContainer.appendChild(messageDiv);
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function updateSidebar(text) {
    const list = document.getElementById('history-list');
    const item = document.createElement('div');
    item.className = 'footer-item';
    item.style.fontSize = '0.85rem';
    item.style.whiteSpace = 'nowrap';
    item.style.overflow = 'hidden';
    item.style.textOverflow = 'ellipsis';
    item.innerHTML = `<i class="ph ph-chat-centered-text"></i> ${text}`;
    list.prepend(item);
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        performTranslation();
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

function scrollToBottom() {
    const chatContainer = document.getElementById('chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function useExample(text) {
    document.getElementById('input-text').value = text;
    autoResize(document.getElementById('input-text'));
    performTranslation();
}

// Performance & Charts
function togglePerformance() {
    const overlay = document.getElementById('perf-overlay');
    overlay.classList.toggle('hidden');
    if (!overlay.classList.contains('hidden')) loadPerformance();
}

async function loadPerformance() {
    try {
        const response = await fetch('/api/performance');
        const data = await response.json();
        if (response.ok) renderCharts(data);
    } catch (e) {
        showToast("Dados indisponíveis.");
    }
}

function renderCharts(history) {
    const evalLabels = history.filter(h => h.eval_loss !== undefined).map(h => `Época ${h.epoch?.toFixed(1) || ''}`);
    const trainLoss = history.map(h => h.loss || null).filter(l => l !== null);
    const evalLoss = history.map(h => h.eval_loss || null).filter(l => l !== null);
    const rouge1 = history.map(h => h.eval_rouge1 || null).filter(r => r !== null);
    
    const lossLabels = history.filter(h => h.loss !== undefined).map(h => `P${Math.round(h.epoch * 10)}`);

    const ctxLoss = document.getElementById('lossChart').getContext('2d');
    if (lossChart) lossChart.destroy();
    
    lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: lossLabels.length > evalLabels.length ? lossLabels : evalLabels,
            datasets: [
                { label: 'Treino', data: trainLoss, borderColor: '#10a37f', borderWidth: 2, pointRadius: 0, fill: false, tension: 0.3 },
                { label: 'Validação', data: '#ef4444', data: evalLoss, borderColor: '#ef4444', borderWidth: 2, pointRadius: 0, fill: false, tension: 0.3 }
            ]
        },
        options: { responsive: true, plugins: { legend: { labels: { color: '#e3e3e3' } } }, scales: { y: { ticks: { color: '#b4b4b4' } }, x: { display: false } } }
    });

    const ctxRouge = document.getElementById('rougeChart').getContext('2d');
    if (rougeChart) rougeChart.destroy();
    
    rougeChart = new Chart(ctxRouge, {
        type: 'line',
        data: {
            labels: evalLabels,
            datasets: [{ label: 'ROUGE-1', data: rouge1, borderColor: '#5436da', borderWidth: 2, fill: false }]
        },
        options: { responsive: true, plugins: { legend: { labels: { color: '#e3e3e3' } } }, scales: { y: { ticks: { color: '#b4b4b4' } }, x: { display: false } } }
    });
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 3000);
}

