/**
 * RLHF Preference Collection - Frontend Logic (Streaming Version with Navigation)
 */

// KaTeX Math Rendering
function renderMath(element) {
    if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(element, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '\\[', right: '\\]', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\(', right: '\\)', display: false}
            ],
            throwOnError: false
        });
    }
}

// Markdown Rendering
function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        return marked.parse(text);
    }
    return text;
}

// State
let currentPrompt = null;
let isGenerating = false;
let generatedResponses = {};
let temperatures = [];
let currentPromptIndex = 0;
let totalPrompts = 0;
let previousChosenIdx = null;
let previousRejectedIdx = null;
let isCompleted = false;
let lastChosenIdx = null;  // Tracks last selection even after deselection

// DOM Elements
const elements = {
    loadingState: document.getElementById('loadingState'),
    completionState: document.getElementById('completionState'),
    mainContent: document.getElementById('mainContent'),
    progressFill: document.getElementById('progressFill'),
    progressPercent: document.getElementById('progressPercent'),
    progressCount: document.getElementById('progressCount'),
    savedCount: document.getElementById('savedCount'),
    remainingCount: document.getElementById('remainingCount'),
    promptBox: document.getElementById('promptBox'),
    responsesContainer: document.getElementById('responsesContainer'),
    skipBtn: document.getElementById('skipBtn'),
    prevBtn: document.getElementById('prevBtn'),
    nextBtn: document.getElementById('nextBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    saveExitBtn: document.getElementById('saveExitBtn'),
    finalDownloadBtn: document.getElementById('finalDownloadBtn'),
    finalSaved: document.getElementById('finalSaved'),
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toastMessage'),
    promptIndicator: document.getElementById('promptIndicator')
};

// API Functions
async function fetchStatus() {
    try {
        const response = await fetch('/api/status');
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch status:', error);
        return null;
    }
}

async function fetchNextPrompt() {
    try {
        const response = await fetch('/api/next');
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch next prompt:', error);
        return null;
    }
}

async function fetchPrevPrompt() {
    try {
        const response = await fetch('/api/prev', { method: 'POST' });
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch prev prompt:', error);
        return null;
    }
}

async function navigateToNextPrompt() {
    try {
        const response = await fetch('/api/next', { method: 'POST' });
        return await response.json();
    } catch (error) {
        console.error('Failed to navigate to next prompt:', error);
        return null;
    }
}

async function savePreference(chosenIdx, rejectedIdx) {
    try {
        const response = await fetch('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chosen_idx: chosenIdx, rejected_idx: rejectedIdx })
        });
        return await response.json();
    } catch (error) {
        console.error('Failed to save preference:', error);
        return null;
    }
}

async function skipCurrent() {
    try {
        const response = await fetch('/api/skip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        return await response.json();
    } catch (error) {
        console.error('Failed to skip:', error);
        return null;
    }
}

// UI Functions
function showToast(message, type = 'success') {
    elements.toastMessage.textContent = message;
    elements.toast.className = 'toast ' + type;
    elements.toast.classList.add('show');
    
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 2000);
}

function updateProgress(status) {
    const total = status.total;
    const completed = status.completed;
    const remaining = status.remaining;
    const percent = total > 0 ? Math.round((completed / total) * 100) : 0;
    
    totalPrompts = total;
    
    elements.progressFill.style.width = percent + '%';
    elements.progressPercent.textContent = percent + '%';
    elements.progressCount.textContent = `(${completed} / ${total})`;
    elements.savedCount.textContent = completed;
    elements.remainingCount.textContent = remaining;
}

function updatePromptIndicator() {
    const promptInput = document.getElementById('promptInput');
    const promptTotal = document.getElementById('promptTotal');
    if (promptInput) {
        promptInput.value = currentPromptIndex + 1;
        promptInput.max = totalPrompts;
    }
    if (promptTotal) {
        promptTotal.textContent = `/ ${totalPrompts}`;
    }
}

function updateNavButtons() {
    if (elements.prevBtn) {
        elements.prevBtn.disabled = currentPromptIndex <= 0 || isGenerating;
    }
    if (elements.nextBtn) {
        elements.nextBtn.disabled = currentPromptIndex >= totalPrompts - 1 || isGenerating;
    }
}

function showLoading() {
    elements.loadingState.style.display = 'flex';
    elements.mainContent.style.display = 'none';
    elements.completionState.style.display = 'none';
}

function showCompletion(status) {
    elements.loadingState.style.display = 'none';
    elements.mainContent.style.display = 'none';
    elements.completionState.style.display = 'block';
    
    elements.finalSaved.textContent = status.completed;
}

function showMain() {
    elements.loadingState.style.display = 'none';
    elements.mainContent.style.display = 'block';
    elements.completionState.style.display = 'none';
}

function createResponseCards(numResponses) {
    elements.responsesContainer.innerHTML = '';
    
    for (let i = 0; i < numResponses; i++) {
        const card = document.createElement('div');
        card.className = 'response-card';
        card.id = `responseCard${i}`;
        
        // For completed prompts: show current selection (not changed yet)
        let badge = '';
        if (isCompleted && previousChosenIdx === i) {
            // Current selection (not changed): Current Chosen + highlight
            badge = '<span class="current-choice-badge">Current Chosen</span>';
            card.classList.add('selected');
        } else if (isCompleted) {
            // Other cards: dimmed
            card.classList.add('dimmed');
        }
        
        card.innerHTML = `
            <div class="response-header">
                <span class="response-label">Response ${String.fromCharCode(65 + i)}</span>
                <span class="response-temp" id="temp${i}"></span>
                ${badge}
                <span class="response-status" id="status${i}">Waiting...</span>
            </div>
            <div class="response-content" id="responseContent${i}">
                <span class="cursor">|</span>
            </div>
            <button class="select-btn" id="selectBtn${i}" disabled onclick="selectResponse(${i})">
                Select ${String.fromCharCode(65 + i)} [${i + 1}]
            </button>
        `;
        elements.responsesContainer.appendChild(card);
    }
}

function displayPrompt(promptData, cachedResponses = {}) {
    currentPrompt = promptData;
    temperatures = promptData.temperatures;
    currentPromptIndex = promptData.prompt_index;
    totalPrompts = promptData.total_prompts;
    generatedResponses = {};
    
    // Store previous selection info
    previousChosenIdx = promptData.chosen_idx;
    previousRejectedIdx = promptData.rejected_idx;
    isCompleted = promptData.completed || false;
    lastChosenIdx = promptData.chosen_idx;  // Initialize lastChosenIdx for this prompt
    
    // Set prompt
    elements.promptBox.textContent = promptData.instruction;
    
    // Create response cards (with previous selection info)
    createResponseCards(temperatures.length);
    
    // Update indicators
    updatePromptIndicator();
    updateNavButtons();
    
    // Disable skip during generation
    if (elements.skipBtn) {
        elements.skipBtn.disabled = true;
    }
    
    // Check for cached responses and display them, or generate new ones
    const cachedKeys = Object.keys(cachedResponses);
    if (cachedKeys.length === temperatures.length) {
        // All responses cached - display them
        displayCachedResponses(cachedResponses);
    } else {
        // Start auto-generation
        generateAllResponses(cachedResponses);
    }
}

function displayCachedResponses(cachedResponses) {
    for (const [tempIdxStr, respData] of Object.entries(cachedResponses)) {
        const tempIdx = parseInt(tempIdxStr);
        const contentEl = document.getElementById(`responseContent${tempIdx}`);
        const statusEl = document.getElementById(`status${tempIdx}`);
        const tempEl = document.getElementById(`temp${tempIdx}`);
        const cardEl = document.getElementById(`responseCard${tempIdx}`);
        
        if (contentEl && statusEl && tempEl && cardEl) {
            // Build temp string with min and avg if available
            let tempStr = `(temp: ${respData.temperature}`;
            if (respData.min_temp_used !== undefined && respData.min_temp_used !== null) {
                tempStr += `, min: ${respData.min_temp_used.toFixed(2)}`;
            }
            if (respData.avg_temp_used !== undefined && respData.avg_temp_used !== null) {
                tempStr += `, avg: ${respData.avg_temp_used.toFixed(2)}`;
            }
            tempStr += ')';
            tempEl.textContent = tempStr;
            
            statusEl.textContent = 'Cached';
            statusEl.className = 'response-status complete';
            contentEl.innerHTML = renderMarkdown(respData.text || '(Empty response)');
            renderMath(contentEl);
            cardEl.classList.add('complete');
            
            generatedResponses[tempIdx] = respData.text;
        }
    }
    
    enableSelection();
    if (elements.skipBtn) {
        elements.skipBtn.disabled = false;
    }
}

async function generateResponse(index, temperature) {
    const contentEl = document.getElementById(`responseContent${index}`);
    const statusEl = document.getElementById(`status${index}`);
    const tempEl = document.getElementById(`temp${index}`);
    const cardEl = document.getElementById(`responseCard${index}`);
    
    // Update UI
    tempEl.textContent = `(temp: ${temperature})`;
    statusEl.textContent = 'Generating...';
    statusEl.className = 'response-status generating';
    contentEl.innerHTML = '<span class="cursor blink">|</span>';
    cardEl.classList.add('generating');
    
    return new Promise((resolve, reject) => {
        const source = new EventSource(`/api/generate?temp_index=${index}&temperature=${temperature}`);
        let fullText = '';
        
        source.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.error) {
                    statusEl.textContent = 'Error';
                    statusEl.className = 'response-status error';
                    cardEl.classList.remove('generating');
                    source.close();
                    reject(new Error(data.error));
                    return;
                }
                
                if (data.done) {
                    // Generation complete
                    statusEl.textContent = 'Complete';
                    statusEl.className = 'response-status complete';
                    cardEl.classList.remove('generating');
                    cardEl.classList.add('complete');
                    
                    // Update temperature display with cur, min and avg
                    let finalTempStr = `(temp: ${temperature}, cur: ${temperature.toFixed(2)}`;
                    if (data.min_temp_used !== undefined && data.min_temp_used < temperature) {
                        finalTempStr += `, min: ${data.min_temp_used.toFixed(2)}`;
                    }
                    if (data.avg_temp_used !== undefined) {
                        finalTempStr += `, avg: ${data.avg_temp_used.toFixed(2)}`;
                    }
                    finalTempStr += ')';
                    tempEl.textContent = finalTempStr;
                    
                    // Remove cursor and render markdown
                    contentEl.innerHTML = renderMarkdown(fullText || '(Empty response)');
                    
                    // Render LaTeX math
                    renderMath(contentEl);
                    
                    // Store response
                    generatedResponses[index] = fullText;
                    
                    source.close();
                    resolve(fullText);
                } else if (data.token) {
                    // Append token
                    fullText += data.token;
                    contentEl.innerHTML = fullText + '<span class="cursor blink">|</span>';
                    
                    // Update temp display in real-time
                    if (data.current_temp !== undefined) {
                        if (data.min_temp_used < temperature) {
                            tempEl.textContent = `(temp: ${temperature}, cur: ${data.current_temp.toFixed(2)}, min: ${data.min_temp_used.toFixed(2)})`;
                        } else {
                            tempEl.textContent = `(temp: ${temperature}, cur: ${data.current_temp.toFixed(2)})`;
                        }
                    }
                    
                    // Auto-scroll
                    contentEl.scrollTop = contentEl.scrollHeight;
                }
            } catch (e) {
                console.error('Parse error:', e);
            }
        };
        
        source.onerror = (error) => {
            statusEl.textContent = 'Connection Error';
            statusEl.className = 'response-status error';
            cardEl.classList.remove('generating');
            source.close();
            reject(error);
        };
    });
}

async function generateAllResponses(cachedResponses = {}) {
    if (isGenerating || !currentPrompt) return;
    
    isGenerating = true;
    updateNavButtons();
    
    try {
        // Generate responses sequentially, skipping cached ones
        for (let i = 0; i < temperatures.length; i++) {
            // Check if already cached
            if (cachedResponses[i.toString()] || cachedResponses[i]) {
                const cached = cachedResponses[i.toString()] || cachedResponses[i];
                const contentEl = document.getElementById(`responseContent${i}`);
                const statusEl = document.getElementById(`status${i}`);
                const tempEl = document.getElementById(`temp${i}`);
                const cardEl = document.getElementById(`responseCard${i}`);
                
                // Build temp string with min and avg if available
                let tempStr = `(temp: ${cached.temperature}`;
                if (cached.min_temp_used !== undefined && cached.min_temp_used !== null) {
                    tempStr += `, min: ${cached.min_temp_used.toFixed(2)}`;
                }
                if (cached.avg_temp_used !== undefined && cached.avg_temp_used !== null) {
                    tempStr += `, avg: ${cached.avg_temp_used.toFixed(2)}`;
                }
                tempStr += ')';
                tempEl.textContent = tempStr;
                
                statusEl.textContent = 'Cached';
                statusEl.className = 'response-status complete';
                contentEl.innerHTML = renderMarkdown(cached.text || '(Empty response)');
                renderMath(contentEl);
                cardEl.classList.add('complete');
                
                generatedResponses[i] = cached.text;
            } else {
                await generateResponse(i, temperatures[i]);
            }
        }
        
        // All done - enable selection
        showToast('Responses ready! Select your preference.', 'success');
        enableSelection();
        
    } catch (error) {
        showToast('Generation failed: ' + error.message, 'error');
    }
    
    isGenerating = false;
    if (elements.skipBtn) {
        elements.skipBtn.disabled = false;
    }
    updateNavButtons();
}

function enableSelection() {
    // Enable all select buttons
    for (let i = 0; i < temperatures.length; i++) {
        const btn = document.getElementById(`selectBtn${i}`);
        if (btn) {
            btn.disabled = false;
        }
    }
}

function disableSelection() {
    for (let i = 0; i < temperatures.length; i++) {
        const btn = document.getElementById(`selectBtn${i}`);
        if (btn) {
            btn.disabled = true;
        }
    }
}

function updateCardSelection(chosenIdx) {
    for (let i = 0; i < temperatures.length; i++) {
        const card = document.getElementById(`responseCard${i}`);
        const headerEl = card.querySelector('.response-header');
        
        // Remove existing badges
        const existingBadge = headerEl.querySelector('.previous-choice-badge, .current-choice-badge');
        if (existingBadge) existingBadge.remove();
        
        if (i === chosenIdx) {
            // Current selection: badge + highlight
            card.classList.remove('dimmed');
            card.classList.add('selected');
            const badge = document.createElement('span');
            badge.className = 'current-choice-badge';
            badge.textContent = 'Current Chosen';
            headerEl.appendChild(badge);
        } else if (i === lastChosenIdx && lastChosenIdx !== null && lastChosenIdx !== chosenIdx) {
            // Previous selection (last chosen): Previously Chosen + dimmed
            card.classList.remove('selected');
            card.classList.add('dimmed');
            const badge = document.createElement('span');
            badge.className = 'previous-choice-badge';
            badge.textContent = 'Previously Chosen';
            headerEl.appendChild(badge);
        } else {
            // Others: dimmed (no badge)
            card.classList.remove('selected');
            card.classList.add('dimmed');
        }
    }
    
    // Update tracking
    lastChosenIdx = chosenIdx;
    previousChosenIdx = chosenIdx;
}

async function selectResponse(chosenIdx) {
    if (isGenerating) return;
    
    // Toggle: same card selected again -> clear selection
    if (previousChosenIdx === chosenIdx) {
        await clearCurrentSelection();
        return;
    }
    
    // Find a rejected response (any other one)
    let rejectedIdx = -1;
    for (let i = 0; i < temperatures.length; i++) {
        if (i !== chosenIdx && generatedResponses[i] !== undefined) {
            rejectedIdx = i;
            break;
        }
    }
    
    if (rejectedIdx === -1) {
        showToast('Need at least 2 responses to compare', 'error');
        return;
    }
    
    disableSelection();
    
    // Visual feedback - update all cards
    updateCardSelection(chosenIdx);
    
    // Save preference
    const result = await savePreference(chosenIdx, rejectedIdx);
    
    if (result && result.success) {
        updateProgress(result.status);
        showToast('Preference saved!', 'success');
        
        // Stay on current prompt (no auto-navigate)
        enableSelection();
    } else {
        showToast('Failed to save', 'error');
        enableSelection();
    }
}

async function skipPrompt() {
    if (isGenerating) return;
    
    disableSelection();
    
    const result = await skipCurrent();
    
    if (result && result.success) {
        updateProgress(result.status);
        showToast('Skipped', 'success');
        
        // Load next prompt
        await loadNextPrompt();
    } else {
        showToast('Failed to skip', 'error');
        enableSelection();
    }
}

async function goBack() {
    if (isGenerating || currentPromptIndex <= 0) return;
    
    showLoading();
    
    const result = await fetchPrevPrompt();
    
    if (result && result.success && result.prompt) {
        showMain();
        displayPrompt(result.prompt, result.cached_responses || {});
        updateProgress(result.status);
    } else {
        showToast('Cannot go back', 'error');
        showMain();
    }
}

async function goNext() {
    if (isGenerating || currentPromptIndex >= totalPrompts - 1) return;
    
    showLoading();
    
    const result = await navigateToNextPrompt();
    
    if (result && result.success && result.prompt) {
        showMain();
        displayPrompt(result.prompt, result.cached_responses || {});
        updateProgress(result.status);
    } else if (result && !result.success) {
        // Can't go next (at end)
        showToast('Already at last prompt', 'info');
        showMain();
    } else {
        showToast('Failed to navigate', 'error');
        showMain();
    }
}

async function goToPrompt(index) {
    if (isGenerating || index < 0 || index >= totalPrompts) return;
    if (index === currentPromptIndex) return;
    
    showLoading();
    
    try {
        const response = await fetch('/api/goto', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index })
        });
        const result = await response.json();
        
        if (result && result.success && result.prompt) {
            showMain();
            displayPrompt(result.prompt, result.cached_responses || {});
            updateProgress(result.status);
        } else {
            showToast('Failed to navigate', 'error');
            showMain();
        }
    } catch (error) {
        console.error('Failed to go to prompt:', error);
        showToast('Failed to navigate', 'error');
        showMain();
    }
}

async function loadNextPrompt() {
    showLoading();
    
    const data = await fetchNextPrompt();
    
    if (!data) {
        showToast('Failed to load next prompt', 'error');
        return;
    }
    
    if (data.done) {
        showCompletion(data.status);
        updateProgress(data.status);
    } else {
        showMain();
        displayPrompt(data.prompt, data.cached_responses || {});
    }
}

function downloadPreferences() {
    window.location.href = '/api/download';
}

function saveAndExit() {
    showToast('Session saved. You can close this tab.', 'success');
    setTimeout(() => {
        alert('Session saved!\n\nTo resume later, just run:\npython rlhf_collect.py\n\nYou can close this tab now.');
    }, 500);
}

// Keyboard Shortcuts
document.addEventListener('keydown', (event) => {
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
        return;
    }
    
    // Ctrl+S: Save & Exit
    if (event.ctrlKey && event.key === 's') {
        event.preventDefault();
        saveAndExit();
        return;
    }
    
    // Arrow Left or B: Go back
    if (event.key === 'ArrowLeft' || event.key === 'b' || event.key === 'B') {
        if (!event.ctrlKey) {
            goBack();
            return;
        }
    }
    
    // Arrow Right or N: Go next
    if (event.key === 'ArrowRight' || event.key === 'n' || event.key === 'N') {
        goNext();
        return;
    }
    
    // S: Skip
    if ((event.key === 's' || event.key === 'S') && !event.ctrlKey) {
        skipPrompt();
        return;
    }
    
    // Number keys for selection
    if (!isGenerating && Object.keys(generatedResponses).length >= 2) {
        const num = parseInt(event.key);
        if (num >= 1 && num <= temperatures.length) {
            selectResponse(num - 1);
        }
    }
});

// Clear Selection Functions
function updateCardDeselection(oldChosenIdx) {
    for (let i = 0; i < temperatures.length; i++) {
        const card = document.getElementById(`responseCard${i}`);
        const headerEl = card.querySelector('.response-header');
        
        // Remove existing badges
        const existingBadge = headerEl.querySelector('.previous-choice-badge, .current-choice-badge');
        if (existingBadge) existingBadge.remove();
        
        // All cards back to normal (not dimmed, not selected)
        card.classList.remove('selected', 'dimmed');
        
        // Add Previously Chosen badge to old selection (no dimming)
        if (i === oldChosenIdx) {
            const badge = document.createElement('span');
            badge.className = 'previous-choice-badge';
            badge.textContent = 'Previously Chosen';
            headerEl.appendChild(badge);
        }
    }
}

async function clearCurrentSelection() {
    if (previousChosenIdx === null) return;
    
    const oldChosenIdx = previousChosenIdx;
    
    try {
        const response = await fetch('/api/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index: currentPromptIndex })
        });
        const result = await response.json();
        
        if (result.success) {
            showToast('Selection cleared', 'success');
            updateProgress(result.status);
            
            // Update local state
            previousChosenIdx = null;
            previousRejectedIdx = null;
            isCompleted = false;
            
            // Update UI with animation (no full refresh)
            updateCardDeselection(oldChosenIdx);
        } else {
            showToast('Failed to clear selection', 'error');
        }
    } catch (error) {
        console.error('Failed to clear selection:', error);
        showToast('Failed to clear selection', 'error');
    }
}

async function clearAllSelections() {
    if (!confirm('Clear all selections? This cannot be undone.')) return;
    
    try {
        const response = await fetch('/api/clear_all', { method: 'POST' });
        const result = await response.json();
        
        if (result.success) {
            showToast('All selections cleared', 'success');
            updateProgress(result.status);
            
            // Update local state
            previousChosenIdx = null;
            previousRejectedIdx = null;
            isCompleted = false;
            
            // Refresh display
            if (result.prompt) {
                const cached = await fetchCachedResponses(currentPromptIndex);
                displayPrompt(result.prompt, cached || {});
            }
        } else {
            showToast('Failed to clear all selections', 'error');
        }
    } catch (error) {
        console.error('Failed to clear all selections:', error);
        showToast('Failed to clear all selections', 'error');
    }
}

async function deleteSessionAndExit() {
    if (!confirm('Delete this session and exit? All data will be permanently deleted.')) return;
    
    try {
        const response = await fetch('/api/delete_session', { method: 'POST' });
        const result = await response.json();
        
        if (result.success) {
            showToast('Session deleted', 'success');
            // Close window after short delay
            setTimeout(() => {
                window.close();
                // If window.close() doesn't work (some browsers block it),
                // show a message
                document.body.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100vh;font-size:1.5rem;color:#666;">Session deleted. You can close this tab.</div>';
            }, 500);
        } else {
            showToast('Failed to delete session', 'error');
        }
    } catch (error) {
        console.error('Failed to delete session:', error);
        showToast('Failed to delete session', 'error');
    }
}

async function fetchCachedResponses(promptIndex) {
    try {
        const response = await fetch(`/api/cached?prompt_idx=${promptIndex}&temp_idx=0`);
        const data = await response.json();
        // We need to get all cached responses - let's fetch them properly
        const allCached = {};
        for (let i = 0; i < temperatures.length; i++) {
            const resp = await fetch(`/api/cached?prompt_idx=${promptIndex}&temp_idx=${i}`);
            const d = await resp.json();
            if (d.cached && d.response) {
                allCached[i] = d.response;
            }
        }
        return allCached;
    } catch (error) {
        console.error('Failed to fetch cached responses:', error);
        return {};
    }
}

// Button Event Listeners
if (elements.skipBtn) {
    elements.skipBtn.addEventListener('click', skipPrompt);
}
if (elements.prevBtn) {
    elements.prevBtn.addEventListener('click', goBack);
}
if (elements.nextBtn) {
    elements.nextBtn.addEventListener('click', goNext);
}
if (elements.downloadBtn) {
    elements.downloadBtn.addEventListener('click', downloadPreferences);
}
if (elements.saveExitBtn) {
    elements.saveExitBtn.addEventListener('click', saveAndExit);
}
if (elements.finalDownloadBtn) {
    elements.finalDownloadBtn.addEventListener('click', downloadPreferences);
}

// Prompt number input event
const promptInput = document.getElementById('promptInput');
if (promptInput) {
    promptInput.addEventListener('change', async (e) => {
        const num = parseInt(e.target.value);
        if (num >= 1 && num <= totalPrompts) {
            await goToPrompt(num - 1);
        } else {
            e.target.value = currentPromptIndex + 1;
        }
    });
    
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.target.blur();
        }
    });
}

// Browser close detection - beforeunload
window.addEventListener('beforeunload', () => {
    navigator.sendBeacon('/api/disconnect');
});

// Heartbeat - 10 second interval
setInterval(() => {
    fetch('/api/heartbeat', { method: 'POST' }).catch(() => {});
}, 10000);

// Initialize
async function init() {
    showLoading();
    
    const status = await fetchStatus();
    
    if (status) {
        updateProgress(status);
        await loadNextPrompt();
    } else {
        showToast('Failed to connect to server', 'error');
    }
}

// Start
init();
