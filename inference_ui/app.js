// ============================================
// Inference Chat - Frontend Logic
// ============================================

// State
let currentConversationId = null;
let conversations = [];
let isStreaming = false;
let sidebarCollapsed = false;
let currentAbortController = null;
let currentMessages = [];
let editingIndex = -1;  // -1 = not in edit mode
let pendingFiles = [];  // Attached files array { type: 'image'|'audio', name, data }
let scrollLockUntil = 0;  // Scroll lock release time (timestamp)
let currentStepQuestion = null;  // Step question context { stepNum, tool, stepName, context, previousSteps }

// Detail Panel State
let detailPanelOpen = true;
let detailPanelWidth = 600;  // Default width
let detailPanelData = {
    goal: '',
    steps: [],
    results: [],      // tool_result accumulation
    codes: {},        // Code per Step { stepIndex: { language, code } }
    analysis: '',     // analyze_plan result
    currentStep: 0
};
let currentCodeStep = null;  // Currently selected Step in Code tab
let currentMode = 'agent';   // 'agent' or 'plan'
let graphPopoutWindow = null; // Reference to popped-out graph window
let nodeGraph = null;         // NodeGraph instance
// DOM Elements
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const newChatBtn = document.getElementById('newChatBtn');
const conversationsList = document.getElementById('conversationsList');
const modelName = document.getElementById('modelName');
const messagesContainer = document.getElementById('messagesContainer');
const messagesWrapper = document.getElementById('messagesWrapper');
const welcomeMessage = document.getElementById('welcomeMessage');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const sendIcon = document.getElementById('sendIcon');
const stopIcon = document.getElementById('stopIcon');
const loadingOverlay = document.getElementById('loadingOverlay');
const mainContent = document.querySelector('.main-content');
const scrollToBottomBtn = document.getElementById('scrollToBottomBtn');

// Detail Panel DOM Elements
const detailPanel = document.getElementById('detailPanel');
const detailToggle = document.getElementById('detailToggle');
const detailClose = document.getElementById('detailClose');
const detailResizeHandle = document.getElementById('detailResizeHandle');
const chatArea = document.getElementById('chatArea');

// ============================================
// Initialization
// ============================================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => initializeApp());
} else {
    initializeApp();
}

async function initializeApp() {
    // Load i18n locale
    await initI18n();
    
    // Apply saved theme
    setTheme(getTheme());
    
    // Apply saved background image, blur, and opacity
    applyBgImage();
    applyBgBlur();
    applyBgOpacity();
    
    // Load sidebar state from localStorage (apply initially without transition)
    const savedSidebarState = localStorage.getItem('sidebarCollapsed');
    if (savedSidebarState === 'true') {
        sidebarCollapsed = true;
        sidebar.style.transition = 'none';
        mainContent.style.transition = 'none';
        mainContent.style.marginLeft = '50px';
        sidebar.classList.add('collapsed');
        
        // Completely remove transition property in next frame
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                sidebar.style.removeProperty('transition');
                mainContent.style.removeProperty('transition');
            });
        });
    }
    
    // Restore mode from localStorage
    const savedMode = localStorage.getItem('inferenceMode');
    if (savedMode === 'plan' || savedMode === 'agent') {
        currentMode = savedMode;
    }
    applyModeToggle();
    
    // Detail panel always visible - set initial width
    document.documentElement.style.setProperty('--detail-panel-width', detailPanelWidth + 'px');
    
    // Initialize node graph
    const graphContainer = document.getElementById('nodeGraphContainer');
    if (graphContainer && typeof NodeGraph !== 'undefined') {
        nodeGraph = new NodeGraph(graphContainer);
        graphContainer.addEventListener('graph-layout-changed', () => {
            if (currentConversationId && nodeGraph) {
                localStorage.setItem('graphState-' + currentConversationId, JSON.stringify(nodeGraph.getState()));
            }
        });
    }
    
    // Setup event listeners
    setupEventListeners();
    
    // Load conversations list
    await loadConversations();
    
    // Auto-load the most recent conversation
    if (conversations.length > 0) {
        await loadConversation(conversations[0].id);
    }
    
    // Get model info
    await getModelInfo();
    
    // Auto-resize textarea
    setupTextareaAutoResize();
}

function setupEventListeners() {
    // Sidebar toggle
    sidebarToggle.addEventListener('click', () => toggleSidebar());
    
    // New chat
    newChatBtn.addEventListener('click', () => createNewChat());
    
    // Send message or Stop
    sendBtn.addEventListener('click', () => {
        if (isStreaming) {
            stopGeneration();
        } else {
            sendMessage();
        }
    });
    
    // Enter to send, Backspace to remove tag
    messageInput.addEventListener('keydown', (e) => {
        // Delete tag with Backspace (when input is empty)
        if (e.key === 'Backspace' && messageInput.value === '' && currentStepQuestion) {
            removeStepTag();
            e.preventDefault();
            return;
        }
        
        if (e.key === 'Enter' && !e.shiftKey && !isStreaming) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // File upload
    setupFileUpload();
    
    // Wheel event - scroll lock/unlock (only intentional user scrolling)
    messagesContainer.addEventListener('wheel', (e) => {
        if (e.deltaY < 0) {
            // Scroll up -> infinite lock (until user scrolls to bottom)
            scrollLockUntil = Infinity;
        } else if (e.deltaY > 0) {
            // Scroll down -> unlock if near bottom (reactivate auto-scroll)
            const threshold = 150;  // Within 150px from bottom
            const distanceFromBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight;
            if (distanceFromBottom < threshold) {
                scrollLockUntil = 0;
            }
        }
    });
    
    // Scroll event - show/hide button only (unlock only via wheel)
    messagesContainer.addEventListener('scroll', () => {
        const threshold = 200;
        const distanceFromBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight;
        
        if (distanceFromBottom > threshold) {
            scrollToBottomBtn.classList.add('visible');
        } else {
            scrollToBottomBtn.classList.remove('visible');
            // Unlock only handled in wheel event (distinguish from programmatic scroll)
        }
    });
    
    // Scroll to bottom button click
    scrollToBottomBtn.addEventListener('click', () => {
        scrollLockUntil = 0;
        scrollToBottom(true);
    });
    
    // Make button clear on input area hover
    const inputArea = document.getElementById('dropZone');
    inputArea.addEventListener('mouseenter', () => {
        scrollToBottomBtn.classList.add('input-hover');
    });
    inputArea.addEventListener('mouseleave', () => {
        scrollToBottomBtn.classList.remove('input-hover');
    });
    
    // Mode toggle
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            currentMode = btn.dataset.mode;
            localStorage.setItem('inferenceMode', currentMode);
            applyModeToggle();
        });
    });
    
    // Graph popout button
    const graphPopoutBtn = document.getElementById('graphPopoutBtn');
    if (graphPopoutBtn) {
        graphPopoutBtn.addEventListener('click', openGraphPopout);
    }
    
    // Graph re-run button
    const graphRerunBtn = document.getElementById('graphRerunBtn');
    if (graphRerunBtn) {
        graphRerunBtn.addEventListener('click', rerunPlanFromGraph);
    }
    
    // BroadcastChannel for graph popout communication
    setupGraphChannel();

    // Remove init loader and show app
    const appInitLoader = document.getElementById('appInitLoader');
    if (appInitLoader) appInitLoader.remove();
    const appContainer = document.getElementById('appContainer');
    if (appContainer) appContainer.style.display = '';
}

// ============================================
// File Upload (Drag & Drop)
// ============================================

function setupFileUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const attachBtn = document.getElementById('attachBtn');
    const chatContainer = document.querySelector('.main-content');
    
    // Helper functions for drag events
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
        chatContainer.classList.add('drag-over');
    }
    
    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        // Only remove if leaving the container entirely
        if (!chatContainer.contains(e.relatedTarget)) {
            dropZone.classList.remove('drag-over');
            chatContainer.classList.remove('drag-over');
        }
    }
    
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
        chatContainer.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    }
    
    // Drag and drop events on input area
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    
    // Also allow drag and drop on messages container
    messagesContainer.addEventListener('dragover', handleDragOver);
    messagesContainer.addEventListener('dragleave', handleDragLeave);
    messagesContainer.addEventListener('drop', handleDrop);
    
    // Click to attach
    attachBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
        fileInput.value = '';  // Reset for same file selection
    });
    
    // Paste from clipboard (Ctrl+V)
    document.addEventListener('paste', (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        
        const files = [];
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                const file = item.getAsFile();
                if (file) files.push(file);
            }
        }
        
        if (files.length > 0) {
            e.preventDefault();
            handleFiles(files);
        }
    });
}

function getMaxAttachments() {
    return parseInt(localStorage.getItem('maxAttachments') || '5');
}

async function handleFiles(files) {
    const maxAttachments = getMaxAttachments();
    
    for (const file of files) {
        if (file.type.startsWith('image/')) {
            const currentCount = pendingFiles.filter(f => f.type === 'image' || f.type === 'document').length;
            if (currentCount >= maxAttachments) {
                alert(`Maximum ${maxAttachments} attachments allowed. You can change this in Settings.`);
                return;
            }
        }
        
        const dataUrl = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.readAsDataURL(file);
        });

        try {
            const resp = await fetch('/api/data/upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: file.name, data: dataUrl })
            });
            if (!resp.ok) throw new Error('Upload failed');
            const result = await resp.json();

            if (file.type.startsWith('image/')) {
                const currentCount = pendingFiles.filter(f => f.type === 'image' || f.type === 'document').length;
                if (currentCount < maxAttachments) {
                    pendingFiles.push({
                        type: 'image', name: result.fileName,
                        uploadId: result.uploadId,
                        previewUrl: `/uploads/${result.uploadId}`
                    });
                }
            } else if (file.type.startsWith('audio/')) {
                pendingFiles.push({
                    type: 'audio', name: result.fileName,
                    uploadId: result.uploadId, data: dataUrl
                });
            } else {
                const currentCount = pendingFiles.filter(f => f.type === 'image' || f.type === 'document').length;
                if (currentCount < maxAttachments) {
                    pendingFiles.push({
                        type: 'document', name: result.fileName,
                        uploadId: result.uploadId,
                        textContent: result.textContent,
                        fileSize: result.fileSize,
                        extractionMethod: result.extractionMethod
                    });
                }
            }
            renderFilePreviews();
        } catch (err) {
            console.error('File upload failed:', err);
            alert(`Failed to upload ${file.name}`);
        }
    }
}

function renderFilePreviews() {
    const container = document.getElementById('filePreviewContainer');
    if (pendingFiles.length === 0) {
        container.innerHTML = '';  // Auto-hidden via :empty CSS
        return;
    }
    
    container.innerHTML = pendingFiles.map((f, i) => `
        <div class="file-preview ${f.type}">
            ${f.type === 'image'
                ? `<img src="${f.previewUrl || f.data}" alt="${escapeHtml(f.name)}">`
                : f.type === 'document'
                ? `<span class="file-icon doc-icon"></span><span class="file-name">${escapeHtml(f.name)}</span>`
                : `<span class="file-icon audio-icon"></span><span class="file-name">${escapeHtml(f.name)}</span>`
            }
            <button class="file-remove" onclick="removeFile(${i})" title="Remove">\u00d7</button>
        </div>
    `).join('');
}

function removeFile(index) {
    pendingFiles.splice(index, 1);
    renderFilePreviews();
}

function setupTextareaAutoResize() {
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
    });
}

// ============================================
// Sidebar
// ============================================

function toggleSidebar(forceCollapse = null) {
    if (forceCollapse !== null) {
        sidebarCollapsed = forceCollapse;
    } else {
        sidebarCollapsed = !sidebarCollapsed;
    }
    
    // Set main area margin-left
    const sidebarWidth = sidebarCollapsed ? 50 : 280;
    mainContent.style.marginLeft = sidebarWidth + 'px';
    
    // Toggle class (transform handled in CSS)
    sidebar.classList.toggle('collapsed', sidebarCollapsed);
    localStorage.setItem('sidebarCollapsed', sidebarCollapsed);
}

// ============================================
// Conversations
// ============================================

async function loadConversations() {
    try {
        const response = await fetch('/api/conversations');
        if (!response.ok) throw new Error('Failed to load conversations');
        
        conversations = await response.json();
        renderConversationsList();
    } catch (error) {
        console.error('Error loading conversations:', error);
        conversations = [];
        renderConversationsList();
    }
}

function renderConversationsList() {
    if (conversations.length === 0) {
        conversationsList.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">💬</div>
                <p>No conversations yet</p>
            </div>
        `;
        return;
    }
    
    conversationsList.innerHTML = conversations.map(conv => `
        <div class="conversation-item ${conv.id === currentConversationId ? 'active' : ''}" 
             data-id="${conv.id}"
             onclick="loadConversation('${conv.id}')">
            <div class="conversation-icon">💬</div>
            <div class="conversation-info">
                <div class="conversation-title">${escapeHtml(conv.title || 'New Chat')}</div>
                <div class="conversation-date">${formatDate(conv.updated_at || conv.created_at)}</div>
            </div>
            <button class="conversation-delete" onclick="deleteConversation(event, '${conv.id}')" title="Delete">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6h14"/>
                </svg>
            </button>
        </div>
    `).join('');
}

async function loadConversation(id) {
    // Stop generation when switching chats (content so far is auto-saved on server)
    if (isStreaming && currentConversationId !== id) {
        stopGeneration();
        // Brief wait to let server save partial response
        await new Promise(r => setTimeout(r, 300));
    }
    
    // Activate auto-scroll on chat load (default behavior)
    scrollLockUntil = 0;
    
    try {
        const response = await fetch(`/api/conversation/${id}`);
        if (!response.ok) throw new Error('Failed to load conversation');
        
        const conversation = await response.json();
        currentConversationId = id;
        
        renderConversationsList();
        renderMessages(conversation.messages || []);
        
        // Restore detail panel from last PLAN_COMPLETE message (or reset if none)
        restoreDetailPanelFromMessages(conversation.messages || []);
    } catch (error) {
        console.error('Error loading conversation:', error);
    }
}

async function createNewChat() {
    // Hide detail panel for new chat
    hideDetailPanel();
    
    // Stop if generating (content so far is auto-saved on server)
    if (isStreaming) {
        stopGeneration();
        await new Promise(r => setTimeout(r, 300));
    }
    
    // Activate auto-scroll for new chat
    scrollLockUntil = 0;
    
    try {
        const response = await fetch('/api/new', { method: 'POST' });
        if (!response.ok) throw new Error('Failed to create new chat');
        
        const data = await response.json();
        currentConversationId = data.id;
        
        await loadConversations();
        renderMessages([]);
        messageInput.focus();
    } catch (error) {
        console.error('Error creating new chat:', error);
    }
}

async function deleteConversation(event, id) {
    event.stopPropagation();
    
    if (!confirm('Delete this conversation?')) return;
    
    try {
        const response = await fetch(`/api/conversation/${id}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed to delete conversation');
        
        if (currentConversationId === id) {
            hideDetailPanel();
            currentConversationId = null;
            renderMessages([]);
        }
        
        await loadConversations();
    } catch (error) {
        console.error('Error deleting conversation:', error);
    }
}

async function clearCurrentChat() {
    if (!currentConversationId) return;
    if (!confirm('Clear all messages in this conversation?')) return;
    
    // Hide detail panel when clearing chat
    hideDetailPanel();
    
    try {
        const response = await fetch(`/api/conversation/${currentConversationId}/clear`, { method: 'POST' });
        if (!response.ok) throw new Error('Failed to clear chat');
        
        renderMessages([]);
        await loadConversations();
    } catch (error) {
        console.error('Error clearing chat:', error);
    }
}

// ============================================
// Messages
// ============================================

function renderMessages(messages) {
    currentMessages = messages;
    
    if (messages.length === 0) {
        welcomeMessage.style.display = 'block';
        messagesWrapper.innerHTML = '';
        messagesWrapper.appendChild(welcomeMessage);
        scrollToBottomBtn.classList.remove('visible');
        return;
    }
    
    welcomeMessage.style.display = 'none';
    messagesWrapper.innerHTML = messages.map((msg, index) => createMessageHTML(msg, index)).join('');
    
    // Render math
    renderMath();
    
    // Scroll to bottom
    scrollToBottom();
}

function createMessageHTML(message, index = -1) {
    const isUser = message.role === 'user';
    const userDisplayName = getUserDisplayName();
    const avatar = isUser ? userDisplayName.charAt(0).toUpperCase() : 'A';
    const roleName = isUser ? userDisplayName : 'Assistant';
    
    let contentHTML = '';
    
    if (!isUser) {
        // Assistant message (including empty placeholder for streaming)
        // Parse all special tokens from content
        const parsed = parseSpecialTokens(message.content || '');
        
        // Render special tokens in order
        if (parsed.think) {
            contentHTML += createCoTHTML(parsed.think);
        }
        if (parsed.tools) {
            contentHTML += createToolsHTML(parsed.tools);
        }
        if (parsed.toolCalls) {
            contentHTML += createToolCallsHTML(parsed.toolCalls);
        }
        if (parsed.toolResults) {
            contentHTML += createToolResultsHTML(parsed.toolResults);
        }
        if (parsed.toolContent) {
            contentHTML += createToolContentHTML(parsed.toolContent);
        }
        if (parsed.fim) {
            contentHTML += createFimHTML(parsed.fim);
        }
        if (parsed.args) {
            contentHTML += createArgsHTML(parsed.args);
        }
        if (parsed.callId) {
            contentHTML += createCallIdBadge(parsed.callId);
        }
        if (parsed.planComplete) {
            contentHTML += createCompletedPlanHTML(parsed.planComplete);
        }
        
        // Render main answer with image/audio placeholders
        let answerHTML = renderMarkdown(parsed.answer);
        answerHTML = answerHTML.replace(/\{\{IMG_PLACEHOLDER\}\}/g, createImgPlaceholder());
        answerHTML = answerHTML.replace(/\{\{AUDIO_PLACEHOLDER\}\}/g, createAudioPlaceholder());
        
        contentHTML += `<div class="answer-content">${answerHTML}</div>`;
    } else {
        // User message - render attached files first
        let filesHTML = '';
        if (message.files && message.files.length > 0) {
            filesHTML = '<div class="message-files">';
            for (const f of message.files) {
                if (f.type === 'image') {
                    filesHTML += `<img src="${f.data}" alt="${escapeHtml(f.name)}" class="message-image" title="${escapeHtml(f.name)}">`;
                } else if (f.type === 'audio') {
                    filesHTML += `<div class="message-audio"><span class="audio-icon">🎵</span><span class="audio-name">${escapeHtml(f.name)}</span></div>`;
                }
            }
            filesHTML += '</div>';
        }
        
        // Extract text content (remove file references like [Image: xxx])
        let textContent = message.content || '';
        textContent = textContent.replace(/\[Image: [^\]]+\]\s*/g, '').replace(/\[Audio: [^\]]+\]\s*/g, '').trim();
        
        // Render markdown and convert step tag markers
        let renderedContent = renderMarkdown(textContent);
        renderedContent = renderedContent.replace(/\{\{STEP_TAG:(\d+)\}\}/g, '<span class="chat-step-tag">Step $1</span>');
        
        contentHTML = filesHTML + `<div class="answer-content">${renderedContent}</div>`;
    }
    
    // Action buttons (only for saved messages with valid index)
    let actionsHTML = '';
    if (index >= 0) {
        if (isUser) {
            actionsHTML = `
                <div class="message-actions">
                    <button class="message-action-btn" onclick="editMessage(${index})" title="Edit and resend">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                    </button>
                    <button class="message-action-btn" onclick="copyMessage(${index})" title="Copy">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                        </svg>
                    </button>
                    <button class="message-action-btn delete-btn" onclick="deleteFromMessage(${index})" title="Delete from here">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                    </button>
                </div>
            `;
        } else {
            actionsHTML = `
                <div class="message-actions">
                    <button class="message-action-btn" onclick="regenerateFrom(${index})" title="Regenerate">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M23 4v6h-6M1 20v-6h6"/>
                            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                        </svg>
                    </button>
                    <button class="message-action-btn" onclick="copyMessage(${index})" title="Copy">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                        </svg>
                    </button>
                    <button class="message-action-btn delete-btn" onclick="deleteFromMessage(${index})" title="Delete from here">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                    </button>
                </div>
            `;
        }
    }
    
    return `
        <div class="message ${message.role}" data-index="${index}">
            <div class="message-header">
                <div class="message-avatar">${avatar}</div>
                <div class="message-role">${roleName}</div>
            </div>
            <div class="message-content">
                ${contentHTML}
            </div>
            ${actionsHTML}
        </div>
    `;
}

// ============================================
// Special Token Parsing
// ============================================

/**
 * Extract a complete JSON object or array from text starting at a given position.
 * Handles nested structures and strings with escaped characters.
 */
function extractJsonFromPosition(text, startPos) {
    let depth = 0;
    let inString = false;
    let escape = false;
    let start = -1;
    
    for (let i = startPos; i < text.length; i++) {
        const char = text[i];
        
        if (escape) {
            escape = false;
            continue;
        }
        
        if (char === '\\' && inString) {
            escape = true;
            continue;
        }
        
        if (char === '"' && !escape) {
            inString = !inString;
            continue;
        }
        
        if (!inString) {
            if (char === '{' || char === '[') {
                if (depth === 0) start = i;
                depth++;
            } else if (char === '}' || char === ']') {
                depth--;
                if (depth === 0) {
                    return { json: text.substring(start, i + 1), endPos: i + 1 };
                }
            }
        }
    }
    return null;
}

function parseSpecialTokens(content) {
    let result = {
        think: null,
        tools: null,
        toolCalls: null,
        toolResults: null,
        toolContent: null,
        planComplete: null,
        fim: null,
        img: [],
        audio: null,
        args: null,
        callId: null,
        answer: content
    };
    
    // [THINK]...[/THINK]
    const thinkMatch = result.answer.match(/\[THINK\]([\s\S]*?)\[\/THINK\]/);
    if (thinkMatch) {
        result.think = thinkMatch[1].trim();
        result.answer = result.answer.replace(/\[THINK\][\s\S]*?\[\/THINK\]/, '');
    }
    
    // [AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS]
    const toolsMatch = result.answer.match(/\[AVAILABLE_TOOLS\]([\s\S]*?)\[\/AVAILABLE_TOOLS\]/);
    if (toolsMatch) {
        result.tools = toolsMatch[1].trim();
        result.answer = result.answer.replace(/\[AVAILABLE_TOOLS\][\s\S]*?\[\/AVAILABLE_TOOLS\]/, '');
    }
    
    // [TOOL_RESULTS]...[/TOOL_RESULTS]
    const toolResultsMatch = result.answer.match(/\[TOOL_RESULTS\]([\s\S]*?)\[\/TOOL_RESULTS\]/);
    if (toolResultsMatch) {
        result.toolResults = toolResultsMatch[1].trim();
        result.answer = result.answer.replace(/\[TOOL_RESULTS\][\s\S]*?\[\/TOOL_RESULTS\]/, '');
    }
    
    // [TOOL_CALLS]tool_name[ARGS]{json} - parse as complete unit
    const toolCallPattern = /\[TOOL_CALLS\](\w+)\[ARGS\]/;
    const toolCallMatch = result.answer.match(toolCallPattern);
    if (toolCallMatch) {
        const toolName = toolCallMatch[1];
        const argsStartPos = result.answer.indexOf('[ARGS]') + 6;
        const extracted = extractJsonFromPosition(result.answer, argsStartPos);
        
        if (extracted) {
            try {
                const args = JSON.parse(extracted.json);
                result.toolCalls = { name: toolName, arguments: args };
                // Remove the entire tool call from answer
                const toolCallStartPos = result.answer.indexOf('[TOOL_CALLS]');
                const fullMatch = result.answer.substring(toolCallStartPos, extracted.endPos);
                result.answer = result.answer.replace(fullMatch, '');
            } catch (e) {
                // JSON parse failed, fall back to simple extraction
                result.toolCalls = toolCallMatch[1];
                result.answer = result.answer.replace(/\[TOOL_CALLS\][\s\S]*?(?=\[(?!\/)|$)/, '');
            }
        } else {
            // Could not extract JSON, fall back to simple extraction
            result.toolCalls = toolCallMatch[1];
            result.answer = result.answer.replace(/\[TOOL_CALLS\][\s\S]*?(?=\[(?!\/)|$)/, '');
        }
    } else {
        // Fallback: [TOOL_CALLS] without [ARGS] format
        const simpleToolCallsMatch = result.answer.match(/\[TOOL_CALLS\]([\s\S]*?)(?=\[(?!\/)|$)/);
        if (simpleToolCallsMatch) {
            result.toolCalls = simpleToolCallsMatch[1].trim();
            result.answer = result.answer.replace(/\[TOOL_CALLS\][\s\S]*?(?=\[(?!\/)|$)/, '');
        }
    }
    
    // Handle incomplete tool calls (during streaming)
    // If we see [TOOL_CALLS] but couldn't parse it completely, mark as pending
    if (!result.toolCalls && result.answer.includes('[TOOL_CALLS]')) {
        result.toolCalls = 'pending';
        const toolCallStart = result.answer.indexOf('[TOOL_CALLS]');
        result.answer = result.answer.substring(0, toolCallStart);
    }
    
    // Also handle case where [ARGS] appears but JSON is incomplete
    if (result.answer.includes('[ARGS]')) {
        result.toolCalls = result.toolCalls || 'pending';
        const argsStart = result.answer.indexOf('[ARGS]');
        const toolCallStart = result.answer.indexOf('[TOOL_CALLS]');
        result.answer = result.answer.substring(0, Math.min(
            toolCallStart >= 0 ? toolCallStart : Infinity,
            argsStart
        ));
    }
    
    // [TOOL_CONTENT] (standalone token)
    const toolContentMatch = result.answer.match(/\[TOOL_CONTENT\]([\s\S]*?)(?=\[(?!\/)|$)/);
    if (toolContentMatch) {
        result.toolContent = toolContentMatch[1].trim();
        result.answer = result.answer.replace(/\[TOOL_CONTENT\][\s\S]*?(?=\[(?!\/)|$)/, '');
    }
    
    // FIM: [PREFIX]...[SUFFIX]...[MIDDLE]...
    const prefixMatch = result.answer.match(/\[PREFIX\]([\s\S]*?)(?=\[SUFFIX\]|\[MIDDLE\]|$)/);
    const suffixMatch = result.answer.match(/\[SUFFIX\]([\s\S]*?)(?=\[MIDDLE\]|$)/);
    const middleMatch = result.answer.match(/\[MIDDLE\]([\s\S]*?)$/);
    
    if (prefixMatch || suffixMatch || middleMatch) {
        result.fim = {
            prefix: prefixMatch ? prefixMatch[1] : '',
            suffix: suffixMatch ? suffixMatch[1] : '',
            middle: middleMatch ? middleMatch[1] : ''
        };
        result.answer = result.answer
            .replace(/\[PREFIX\][\s\S]*?(?=\[SUFFIX\]|\[MIDDLE\]|$)/, '')
            .replace(/\[SUFFIX\][\s\S]*?(?=\[MIDDLE\]|$)/, '')
            .replace(/\[MIDDLE\][\s\S]*?$/, '');
    }
    
    // [IMG], [IMG_BREAK], [IMG_END]
    const imgMatches = result.answer.match(/\[IMG\]|\[IMG_BREAK\]|\[IMG_END\]/g);
    if (imgMatches) {
        result.img = imgMatches;
        result.answer = result.answer.replace(/\[IMG\]|\[IMG_BREAK\]|\[IMG_END\]/g, '{{IMG_PLACEHOLDER}}');
    }
    
    // [AUDIO], [BEGIN_AUDIO]
    const audioMatch = result.answer.match(/\[AUDIO\]|\[BEGIN_AUDIO\]/);
    if (audioMatch) {
        result.audio = audioMatch[0];
        result.answer = result.answer.replace(/\[AUDIO\]|\[BEGIN_AUDIO\]/g, '{{AUDIO_PLACEHOLDER}}');
    }
    
    // [ARGS]...(JSON until next token or end) - only if not already consumed by tool call
    if (!result.toolCalls || typeof result.toolCalls === 'string') {
        const argsMatch = result.answer.match(/\[ARGS\]([\s\S]*?)(?=\[(?!\/)|$)/);
        if (argsMatch) {
            result.args = argsMatch[1].trim();
            result.answer = result.answer.replace(/\[ARGS\][\s\S]*?(?=\[(?!\/)|$)/, '');
        }
    }
    
    // [CALL_ID]...(ID until next token or whitespace)
    const callIdMatch = result.answer.match(/\[CALL_ID\](\S+)/);
    if (callIdMatch) {
        result.callId = callIdMatch[1].trim();
        result.answer = result.answer.replace(/\[CALL_ID\]\S+/, '');
    }
    
    // [PLAN_COMPLETE]{json} - completed plan with results
    const planCompleteMatch = result.answer.match(/\[PLAN_COMPLETE\]([\s\S]*)$/);
    if (planCompleteMatch) {
        try {
            result.planComplete = JSON.parse(planCompleteMatch[1].trim());
        } catch (e) {
            // JSON parse failed, keep raw
            result.planComplete = { raw: planCompleteMatch[1].trim() };
        }
        result.answer = result.answer.replace(/\[PLAN_COMPLETE\][\s\S]*$/, '');
    }
    
    result.answer = result.answer.trim();
    return result;
}

// Legacy wrapper for backward compatibility
function parseCoT(content) {
    const result = parseSpecialTokens(content);
    return { cot: result.think, answer: result.answer };
}

// ============================================
// Special Token HTML Renderers
// ============================================

function createCoTHTML(cotContent) {
    const preview = cotContent.substring(0, 80).replace(/\n/g, ' ');
    const displayPreview = preview.length < cotContent.length ? preview + '...' : preview;
    
    return `
        <div class="cot-container">
            <button class="cot-toggle" onclick="toggleSpecialToken(this)">
                <span class="cot-icon">✦</span>
                <span class="cot-label">${t('label.cot')}</span>
                <span class="cot-arrow">▼</span>
            </button>
            <div class="cot-content">${escapeHtml(cotContent)}</div>
        </div>
    `;
}

function createToolsHTML(content) {
    return `
        <div class="special-token-container tools-container">
            <button class="special-toggle" onclick="toggleSpecialToken(this)">
                <span class="special-icon">🔧</span>
                <span class="special-label">Available Tools</span>
                <span class="special-arrow">▶</span>
            </button>
            <div class="special-content"><pre>${escapeHtml(content)}</pre></div>
        </div>
    `;
}

/**
 * Create inline plan steps HTML for rendering create_plan tool calls
 */
function createInlinePlanStepsHTML(args) {
    const goal = args.goal || '';
    const steps = args.steps || [];
    
    // Goal header with plan reference button (same structure as createCompletedPlanHTML)
    let goalHTML = goal ? `
        <div class="plan-goal plan-goal-row">
            <span class="plan-goal-text">${escapeHtml(goal)}</span>
            <button class="plan-ref-btn" onclick="askAboutPlan(this)" title="${t('tooltip.plan_ref')}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            </button>
        </div>
    ` : '';
    
    let stepsHTML = '';
    steps.forEach((step, index) => {
        const stepNum = index + 1;
        stepsHTML += `
            <div class="plan-step pending" data-step-id="${stepNum}" data-tool="${escapeHtml(step.tool || '')}">
                <div class="step-header">
                    <div class="step-header-main" onclick="toggleStepResult(this.parentElement)">
                        <div class="step-indicator">${stepNum}</div>
                        <div class="step-content">
                            <div class="step-name">${escapeHtml(step.name || '')}</div>
                            <div class="step-tool">${escapeHtml(step.tool || '')}</div>
                        </div>
                    </div>
                    <div class="step-actions">
                        <button class="step-action-btn" onclick="event.stopPropagation(); retryStep(${stepNum})" title="${t('tooltip.retry')}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M23 4v6h-6M1 20v-6h6"/>
                                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); editStepResult(${stepNum})" title="${t('tooltip.edit_result')}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); askAboutStep(${stepNum})" title="${t('tooltip.ask')}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                                <line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                        </button>
                    </div>
                    <div class="step-toggle" onclick="toggleStepResult(this.closest('.step-header'))" style="visibility: hidden;">▼</div>
                </div>
                <div class="step-result" style="display: none;"></div>
            </div>
        `;
    });
    
    return `<div class="plan-steps-box">${goalHTML}<div class="plan-steps">${stepsHTML}</div></div>`;
}

/**
 * Create HTML for a completed plan (loaded from saved [PLAN_COMPLETE] message)
 */
function createCompletedPlanHTML(planData) {
    const goal = planData.goal || '';
    const steps = planData.steps || [];
    const results = planData.results || [];
    
    const resultMap = mergeResultsByStep(results);
    
    // Goal header with plan reference button (button inside goal bar)
    let goalHTML = goal ? `
        <div class="plan-goal plan-goal-row">
            <span class="plan-goal-text">${escapeHtml(goal)}</span>
            <button class="plan-ref-btn" onclick="askAboutPlan(this)" title="${t('tooltip.plan_ref')}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            </button>
        </div>
    ` : '';
    
    let stepsHTML = '';
    steps.forEach((step, index) => {
        const stepNum = index + 1;
        const result = resultMap[stepNum];
        const statusClass = result
            ? (result.stopped ? 'stopped' : (result.success ? 'completed' : 'error'))
            : (planData.stopped ? 'stopped' : 'pending');
        const indicator = result
            ? (result.stopped ? '◼' : (result.success ? '✓' : '!'))
            : (planData.stopped ? '◼' : stepNum);
        
        let resultHTML = '';
        if (result) {
            const parts = (result._merged || [result]).map(r => formatStepResult({
                success: r.success,
                thought: r.thought,
                action: r.action,
                error: r.error,
                result: r.result
            }));
            const formattedResult = parts.join('<hr class="tool-result-divider">');
            resultHTML = `<div class="step-result" style="display: block;">${formattedResult}</div>`;
        }
        
        stepsHTML += `
            <div class="plan-step ${statusClass}" data-step-id="${stepNum}" data-tool="${escapeHtml(step.tool || '')}">
                <div class="step-header">
                    <div class="step-header-main" onclick="toggleStepResult(this.parentElement)">
                        <div class="step-indicator">${indicator}</div>
                        <div class="step-content">
                            <div class="step-name">${escapeHtml(step.name || '')}</div>
                            <div class="step-tool">${escapeHtml(step.tool || '')}</div>
                        </div>
                    </div>
                    <div class="step-actions">
                        <button class="step-action-btn" onclick="event.stopPropagation(); retryStep(${stepNum})" title="${t('tooltip.retry')}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M23 4v6h-6M1 20v-6h6"/>
                                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); editStepResult(${stepNum})" title="${t('tooltip.edit_result')}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); askAboutStep(${stepNum})" title="${t('tooltip.ask')}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                                <line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                        </button>
                    </div>
                    <div class="step-toggle" onclick="toggleStepResult(this.closest('.step-header'))" style="visibility: ${result ? 'visible' : 'hidden'};">${result ? '▲' : '▼'}</div>
                </div>
                ${resultHTML}
            </div>
        `;
    });
    
    return `<div class="plan-steps-box completed-plan">${goalHTML}<div class="plan-steps">${stepsHTML}</div></div>`;
}

function createToolCallsHTML(content) {
    // Skip if content is 'pending' (incomplete tool call that failed to parse)
    if (content === 'pending') {
        return '';
    }
    
    // Check if content is a parsed tool call object (from new parsing)
    if (typeof content === 'object' && content.name === 'create_plan') {
        return createInlinePlanStepsHTML(content.arguments);
    }
    
    // Fallback to existing behavior for raw string content or other tools
    const displayContent = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
    return `
        <div class="special-token-container tool-calls-container">
            <div class="special-badge tool-calls-badge">
                <span class="special-icon">⚡</span>
                <span class="special-label">Tool Call</span>
            </div>
            <div class="special-content"><pre>${escapeHtml(displayContent)}</pre></div>
        </div>
    `;
}

function createToolResultsHTML(content) {
    return `
        <div class="special-token-container tool-results-container">
            <button class="special-toggle" onclick="toggleSpecialToken(this)">
                <span class="special-icon">📋</span>
                <span class="special-label">Tool Results</span>
                <span class="special-arrow">▶</span>
            </button>
            <div class="special-content"><pre>${escapeHtml(content)}</pre></div>
        </div>
    `;
}

function createToolContentHTML(content) {
    return `
        <div class="special-token-container tool-content-container">
            <div class="special-badge">
                <span class="special-icon">📦</span>
                <span class="special-label">Tool Content</span>
            </div>
            <div class="special-content"><pre>${escapeHtml(content)}</pre></div>
        </div>
    `;
}

function createFimHTML(fim) {
    let html = '<div class="fim-container">';
    
    if (fim.prefix) {
        html += `<div class="fim-section fim-prefix">
            <span class="fim-label">PREFIX</span>
            <pre>${escapeHtml(fim.prefix)}</pre>
        </div>`;
    }
    
    if (fim.middle) {
        html += `<div class="fim-section fim-middle">
            <span class="fim-label">MIDDLE (Generated)</span>
            <pre>${escapeHtml(fim.middle)}</pre>
        </div>`;
    }
    
    if (fim.suffix) {
        html += `<div class="fim-section fim-suffix">
            <span class="fim-label">SUFFIX</span>
            <pre>${escapeHtml(fim.suffix)}</pre>
        </div>`;
    }
    
    html += '</div>';
    return html;
}

function createImgPlaceholder() {
    return `<span class="img-placeholder" title="Image token">🖼️</span>`;
}

function createAudioPlaceholder() {
    return `<span class="audio-placeholder" title="Audio token">🔊</span>`;
}

function createArgsHTML(content) {
    let formatted = content;
    try {
        const parsed = JSON.parse(content);
        formatted = JSON.stringify(parsed, null, 2);
    } catch (e) {
        // Not valid JSON, use as-is
    }
    return `
        <div class="args-container">
            <span class="args-label">Args:</span>
            <pre class="args-json">${escapeHtml(formatted)}</pre>
        </div>
    `;
}

function createCallIdBadge(id) {
    return `<span class="call-id-badge" title="Call ID">#${escapeHtml(id)}</span>`;
}

function toggleSpecialToken(button) {
    button.classList.toggle('expanded');
}

// Keep old function name for backward compatibility
function toggleCoT(button) {
    toggleSpecialToken(button);
}

// ============================================
// Send Message
// ============================================

async function sendMessage(customContent = null) {
    const content = customContent || messageInput.value.trim();
    const files = [...pendingFiles];  // Copy pending files
    
    if ((!content && files.length === 0) || isStreaming) return;
    
    // Check for step question tag (pill-based)
    const stepTag = document.querySelector('#inputTags .input-tag');
    if (stepTag && currentStepQuestion && content) {
        await sendStepQuestionFromMain(content);
        // Clear tag after sending
        document.getElementById('inputTags').innerHTML = '';
        messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
        return;
    }
    
    // Clear step question context if not a step question
    currentStepQuestion = null;
    document.getElementById('inputTags').innerHTML = '';
    messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
    
    // Activate auto-scroll when sending message
    scrollLockUntil = 0;
    
    // Create new conversation if needed
    if (!currentConversationId) {
        await createNewChat();
    }
    
    // Clear input and files
    if (!customContent) {
        messageInput.value = '';
        messageInput.style.height = 'auto';
    }
    pendingFiles = [];
    renderFilePreviews();
    
    // Build display content for user message
    let displayContent = content;
    if (files.length > 0) {
        const fileLabels = files.map(f => {
            if (f.type === 'image') return `[Image: ${f.name}]`;
            if (f.type === 'document') return `[Document: ${f.name}]`;
            return `[Audio: ${f.name}]`;
        }).join(' ');
        displayContent = fileLabels + (content ? '\n' + content : '');
    }
    
    // Add user message to UI
    appendMessage({ role: 'user', content: displayContent, files });
    
    // Create placeholder for assistant response
    const assistantMessage = appendMessage({ role: 'assistant', content: '' });
    const contentDiv = assistantMessage.querySelector('.message-content');
    
    // Start streaming
    isStreaming = true;
    currentAbortController = new AbortController();
    setStreamingUI(true);
    
    // Add streaming indicator
    contentDiv.innerHTML = '<span class="streaming-indicator"><span class="streaming-dot"></span><span class="streaming-dot"></span><span class="streaming-dot"></span></span>';
    
    let fullContent = '';
    let buffer = '';
    
    try {
        // Build request body with files and mode
        const requestBody = {
            conversation_id: currentConversationId,
            message: content,
            mode: currentMode
        };
        if (files.length > 0) {
            requestBody.files = files.map(f => {
                if (f.type === 'image') return { type: 'image', name: f.name, uploadId: f.uploadId };
                if (f.type === 'document') return { type: 'document', name: f.name, uploadId: f.uploadId, textContent: f.textContent };
                return { type: f.type, name: f.name, data: f.data };
            });
        }
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
            signal: currentAbortController.signal
        });
        
        if (!response.ok) throw new Error('Failed to send message');
        
        // Update sidebar title immediately (server already updated it when saving user message)
        loadConversations();
        
        // Read SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();  // Keep incomplete line for next chunk
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.token) {
                            fullContent += data.token;
                            updateAssistantMessage(contentDiv, fullContent);
                        }
                        
                        // Handle tool call events
                        if (data.tool_call) {
                            if (data.tool_call.name === 'create_plan') {
                                const toolBox = createPlanStepsBox(data.tool_call);
                                contentDiv.appendChild(toolBox);
                                scrollToBottom();
                                
                                // Open Detail Panel with plan data
                                const args = data.tool_call.arguments || {};
                                openDetailPanel({
                                    goal: args.goal || content || '',
                                    userMessage: content,
                                    steps: (args.steps || []).map((s, i) => ({
                                        id: i + 1,
                                        name: s.name || '',
                                        tool: s.tool || '',
                                        description: s.description || ''
                                    }))
                                });
                            } else {
                                // Check if this tool is part of the current plan
                                const planBox = document.getElementById('current-plan-box');
                                if (planBox) {
                                    // Find step by tool name (first pending one with matching tool)
                                    const toolName = data.tool_call.name;
                                    const stepEl = planBox.querySelector(`.plan-step.pending[data-tool="${toolName}"]`) ||
                                                   planBox.querySelector(`.plan-step:not(.completed):not(.error)[data-tool="${toolName}"]`);
                                    if (stepEl && !stepEl.classList.contains('completed')) {
                                        stepEl.classList.remove('pending');
                                        stepEl.classList.add('running');
                                        // Force visual update
                                        void stepEl.offsetHeight;
                                        scrollToBottom();
                                    }
                                }
                                // If not part of plan, don't create separate box (cleaner UI)
                            }
                        }
                        
                        // Handle tool result events
                        if (data.tool_result) {
                            updateToolResultBox(data.tool_result);
                            scrollToBottom();
                            
                            // Update Detail Panel with tool result
                            if (detailPanelOpen && data.tool_result.step !== undefined) {
                                const stepIndex = data.tool_result.step - 1; // Convert to 0-based
                                addToolResultToDetailPanel(stepIndex, data.tool_result);
                            }
                            if (nodeGraph && data.tool_result.step !== undefined) {
                                const status = data.tool_result.success ? 'completed' : 'error';
                                nodeGraph.setNodeStatus(`step-${data.tool_result.step}`, status);
                                broadcastGraphMessage({ type: 'step-update', payload: { step: data.tool_result.step, status } });
                            }
                        }
                        
                        // Handle step start events (LLM starting to generate tool call for this step)
                        if (data.step_start) {
                            const planBox = document.getElementById('current-plan-box');
                            if (planBox) {
                                const stepEl = planBox.querySelector(`[data-step-id="${data.step_start.step}"]`);
                                if (stepEl && !stepEl.classList.contains('completed')) {
                                    stepEl.classList.remove('pending');
                                    stepEl.classList.add('running');
                                    scrollToBottom();
                                }
                            }
                            if (nodeGraph) {
                                nodeGraph.setNodeStatus(`step-${data.step_start.step}`, 'running');
                            }
                            broadcastGraphMessage({ type: 'step-update', payload: { step: data.step_start.step, status: 'running' } });
                        }
                        
                        if (data.done) {
                            // If plan execution completed, update plan box with results
                            if (data.plan_complete) {
                                // Remove streaming indicator
                                const streamingIndicator = contentDiv.querySelector('.streaming-indicator');
                                if (streamingIndicator) {
                                    streamingIndicator.remove();
                                }
                                
                                const planBox = document.getElementById('current-plan-box');
                                if (planBox) {
                                    updatePlanBoxWithResults(planBox, data.plan_complete);
                                }
                                
                                // Trigger Detail Panel completion updates
                                if (detailPanelOpen) {
                                    onPlanComplete();
                                }
                                
                                // Update currentMessages
                                currentMessages.push({ role: 'user', content: displayContent });
                                currentMessages.push({ role: 'assistant', content: fullContent });
                                
                                // Update action buttons on BOTH messages
                                const messageElements = messagesWrapper.querySelectorAll('.message');
                                const userMsgIndex = currentMessages.length - 2;
                                const assistantMsgIndex = currentMessages.length - 1;
                                
                                if (messageElements.length >= 2) {
                                    const userMsgEl = messageElements[messageElements.length - 2];
                                    const assistantMsgEl = messageElements[messageElements.length - 1];
                                    
                                    // Re-render user message
                                    if (userMsgEl) {
                                        userMsgEl.outerHTML = createMessageHTML(currentMessages[userMsgIndex], userMsgIndex);
                                    }
                                    
                                    // Update assistant message action buttons WITHOUT re-rendering (to avoid duplicate plan box)
                                    if (assistantMsgEl) {
                                        // Set data-index attribute
                                        assistantMsgEl.setAttribute('data-index', assistantMsgIndex);
                                        
                                        // Find or create actions div
                                        let actionsDiv = assistantMsgEl.querySelector('.message-actions');
                                        if (!actionsDiv) {
                                            actionsDiv = document.createElement('div');
                                            actionsDiv.className = 'message-actions';
                                            assistantMsgEl.appendChild(actionsDiv);
                                        }
                                        
                                        // Set action buttons HTML for assistant message
                                        actionsDiv.innerHTML = `
                                            <button class="message-action-btn" onclick="regenerateFrom(${assistantMsgIndex})" title="Regenerate">
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                                    <path d="M23 4v6h-6M1 20v-6h6"/>
                                                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                                                </svg>
                                            </button>
                                            <button class="message-action-btn" onclick="copyMessage(${assistantMsgIndex})" title="Copy">
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                                                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                                                </svg>
                                            </button>
                                            <button class="message-action-btn delete-btn" onclick="deleteFromMessage(${assistantMsgIndex})" title="Delete from here">
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                                    <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                                </svg>
                                            </button>
                                        `;
                                    }
                                }
                                
                                // Cancel stream and cleanup
                                await reader.cancel();
                                scrollToBottom();
                                await loadConversations();
                                return;
                            }
                            
                            // Non-plan done handling: Preserve existing tool boxes
                            const existingToolBoxes = contentDiv.querySelectorAll('.plan-steps-box, .tool-call-box');
                            const toolBoxesToPreserve = [];
                            existingToolBoxes.forEach(box => {
                                box.remove();  // Detach from DOM
                                toolBoxesToPreserve.push(box);
                            });
                            
                            // Final update with all special tokens
                            const parsed = parseSpecialTokens(fullContent);
                            let finalHTML = '';
                            
                            if (parsed.think) {
                                finalHTML += createCoTHTML(parsed.think);
                            }
                            if (parsed.tools) {
                                finalHTML += createToolsHTML(parsed.tools);
                            }
                            // Skip toolCalls if we have existing tool boxes (already displayed)
                            if (toolBoxesToPreserve.length === 0 && parsed.toolCalls) {
                                finalHTML += createToolCallsHTML(parsed.toolCalls);
                            }
                            if (parsed.toolResults) {
                                finalHTML += createToolResultsHTML(parsed.toolResults);
                            }
                            if (parsed.toolContent) {
                                finalHTML += createToolContentHTML(parsed.toolContent);
                            }
                            if (parsed.fim) {
                                finalHTML += createFimHTML(parsed.fim);
                            }
                            if (parsed.args) {
                                finalHTML += createArgsHTML(parsed.args);
                            }
                            if (parsed.callId) {
                                finalHTML += createCallIdBadge(parsed.callId);
                            }
                            
                            let answerHTML = renderMarkdown(parsed.answer);
                            answerHTML = answerHTML.replace(/\{\{IMG_PLACEHOLDER\}\}/g, createImgPlaceholder());
                            answerHTML = answerHTML.replace(/\{\{AUDIO_PLACEHOLDER\}\}/g, createAudioPlaceholder());
                            finalHTML += `<div class="answer-content">${answerHTML}</div>`;
                            
                            contentDiv.innerHTML = finalHTML;
                            
                            // Re-append preserved tool boxes at the top
                            toolBoxesToPreserve.forEach(box => {
                                contentDiv.insertBefore(box, contentDiv.firstChild);
                            });
                            
                            renderMath();
                            
                            // Cancel stream and exit loop
                            await reader.cancel();
                            scrollToBottom();
                            // Don't reload conversation - content is already rendered with plan boxes
                            // Only refresh sidebar to update title
                            await loadConversations();
                            
                            // Update currentMessages to fix action buttons (edit/copy/delete/regenerate)
                            // Since we don't call loadConversation, we need to manually update the array
                            currentMessages.push({ role: 'user', content: displayContent });
                            currentMessages.push({ role: 'assistant', content: fullContent });
                            
                            // Update DOM elements with correct indices for action buttons
                            const messageElements = messagesWrapper.querySelectorAll('.message');
                            const userMsgIndex = currentMessages.length - 2;
                            const assistantMsgIndex = currentMessages.length - 1;
                            
                            if (messageElements.length >= 2) {
                                const userMsgEl = messageElements[messageElements.length - 2];
                                const assistantMsgEl = messageElements[messageElements.length - 1];
                                
                                // Re-render user message with correct index
                                if (userMsgEl) {
                                    userMsgEl.outerHTML = createMessageHTML(currentMessages[userMsgIndex], userMsgIndex);
                                }
                                
                                // Re-render assistant message (plan box is already created by createMessageHTML from saved content)
                                if (assistantMsgEl) {
                                    assistantMsgEl.outerHTML = createMessageHTML(currentMessages[assistantMsgIndex], assistantMsgIndex);
                                }
                            }
                            
                            return;
                        }
                        
                        if (data.error) {
                            contentDiv.innerHTML = `<div class="error">Error: ${escapeHtml(data.error)}</div>`;
                            await reader.cancel();
                            return;
                        }
                    } catch (e) {
                        // Skip invalid JSON
                    }
                }
            }
            
            scrollToBottom();
        }
        
        // Reload conversations to update title
        await loadConversations();
        
        // Update currentMessages for fallback case (if data.done wasn't received)
        if (!currentMessages.find(m => m.role === 'user' && m.content === displayContent)) {
            currentMessages.push({ role: 'user', content: displayContent });
            currentMessages.push({ role: 'assistant', content: fullContent });
            
            // Update DOM elements with correct indices for action buttons
            const messageElements = messagesWrapper.querySelectorAll('.message');
            const userMsgIndex = currentMessages.length - 2;
            const assistantMsgIndex = currentMessages.length - 1;
            
            if (messageElements.length >= 2) {
                const userMsgEl = messageElements[messageElements.length - 2];
                const assistantMsgEl = messageElements[messageElements.length - 1];
                
                if (userMsgEl) {
                    userMsgEl.outerHTML = createMessageHTML(currentMessages[userMsgIndex], userMsgIndex);
                }
                
                if (assistantMsgEl) {
                    assistantMsgEl.outerHTML = createMessageHTML(currentMessages[assistantMsgIndex], assistantMsgIndex);
                }
            }
        }
        
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Generation stopped by user');
            const streamingIndicator = contentDiv.querySelector('.streaming-indicator');
            if (streamingIndicator) streamingIndicator.remove();
            const planBox = document.getElementById('current-plan-box');
            if (planBox) {
                planBox.querySelectorAll('.plan-step.running').forEach(step => {
                    step.classList.remove('running');
                    step.classList.add('stopped');
                    const ind = step.querySelector('.step-indicator');
                    if (ind) ind.textContent = '◼';
                });
                planBox.querySelectorAll('.plan-step.pending').forEach(step => {
                    step.classList.remove('pending');
                    step.classList.add('stopped');
                    const ind = step.querySelector('.step-indicator');
                    if (ind) ind.textContent = '◼';
                });
            }
            if (nodeGraph) {
                nodeGraph.resetRunningNodes();
            }

            currentMessages.push({ role: 'user', content: displayContent });
            currentMessages.push({ role: 'assistant', content: fullContent });
            const userMsgIndex = currentMessages.length - 2;
            const assistantMsgIndex = currentMessages.length - 1;

            const messageElements = messagesWrapper.querySelectorAll('.message');
            if (messageElements.length >= 2) {
                const userMsgEl = messageElements[messageElements.length - 2];
                const assistantMsgEl = messageElements[messageElements.length - 1];
                if (userMsgEl) {
                    userMsgEl.setAttribute('data-index', userMsgIndex);
                }
                if (assistantMsgEl) {
                    assistantMsgEl.setAttribute('data-index', assistantMsgIndex);
                    let actionsDiv = assistantMsgEl.querySelector('.message-actions');
                    if (!actionsDiv) {
                        actionsDiv = document.createElement('div');
                        actionsDiv.className = 'message-actions';
                        assistantMsgEl.appendChild(actionsDiv);
                    }
                    actionsDiv.innerHTML = `
                        <button class="message-action-btn" onclick="regenerateFrom(${assistantMsgIndex})" title="Regenerate">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M23 4v6h-6M1 20v-6h6"/>
                                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                            </svg>
                        </button>
                        <button class="message-action-btn" onclick="copyMessage(${assistantMsgIndex})" title="Copy">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                            </svg>
                        </button>
                        <button class="message-action-btn delete-btn" onclick="deleteFromMessage(${assistantMsgIndex})" title="Delete from here">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                            </svg>
                        </button>
                    `;
                }
            }

            await loadConversations();
        } else {
            console.error('Error sending message:', error);
            contentDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        }
    } finally {
        isStreaming = false;
        currentAbortController = null;
        setStreamingUI(false);
        const remainingIndicator = contentDiv.querySelector('.streaming-indicator');
        if (remainingIndicator) remainingIndicator.remove();
    }
}

function appendMessage(message) {
    welcomeMessage.style.display = 'none';
    
    const messageDiv = document.createElement('div');
    messageDiv.innerHTML = createMessageHTML(message);
    const messageElement = messageDiv.firstElementChild;
    messagesWrapper.appendChild(messageElement);
    
    scrollToBottom();
    return messageElement;
}

function updateAssistantMessage(contentDiv, content) {
    // Show streaming content with partial special token parsing
    const parsed = parseSpecialTokens(content);
    
    // Check for existing elements to avoid recreating them
    const existingPlanBox = contentDiv.querySelector('.plan-steps-box');
    const existingAnswerDiv = contentDiv.querySelector('.answer-content');
    const existingStreamingIndicator = contentDiv.querySelector('.streaming-indicator');
    
    // Render main answer with placeholders
    let answerHTML = renderMarkdown(parsed.answer);
    answerHTML = answerHTML.replace(/\{\{IMG_PLACEHOLDER\}\}/g, createImgPlaceholder());
    answerHTML = answerHTML.replace(/\{\{AUDIO_PLACEHOLDER\}\}/g, createAudioPlaceholder());
    
    // If we already have the structure, just update the answer content (reduces flicker)
    if (existingAnswerDiv) {
        existingAnswerDiv.innerHTML = answerHTML;
        
        // Ensure streaming indicator exists
        if (!existingStreamingIndicator) {
            const indicator = document.createElement('span');
            indicator.className = 'streaming-indicator';
            indicator.innerHTML = '<span class="streaming-dot"></span><span class="streaming-dot"></span><span class="streaming-dot"></span>';
            contentDiv.appendChild(indicator);
        }
        return;
    }
    
    // First time setup - create structure
    let html = '';
    
    // Show thinking indicator during streaming
    if (parsed.think) {
        html += `<div class="cot-container">
            <button class="cot-toggle" onclick="toggleSpecialToken(this)">
                <span class="cot-icon">✦</span>
                <span class="cot-label" style="color: var(--text-cot);">${t('label.cot')}</span>
                <span class="cot-arrow">▼</span>
            </button>
        </div>`;
    }
    
    // Show tool calls indicator during streaming (only if no plan box exists)
    if (parsed.toolCalls && !existingPlanBox) {
        html += `<div class="special-badge tool-calls-badge">
            <span class="special-icon">⚡</span>
            <span class="special-label">Tool Call...</span>
        </div>`;
    }
    
    // Show FIM indicator during streaming
    if (parsed.fim) {
        html += `<div class="special-badge">
            <span class="special-icon">📝</span>
            <span class="special-label">Code Generation...</span>
        </div>`;
    }
    
    html += `<div class="answer-content">${answerHTML}</div>`;
    html += '<span class="streaming-indicator"><span class="streaming-dot"></span><span class="streaming-dot"></span><span class="streaming-dot"></span></span>';
    
    // Preserve plan box if it exists
    if (existingPlanBox) {
        existingPlanBox.remove();
        contentDiv.innerHTML = html;
        contentDiv.insertBefore(existingPlanBox, contentDiv.firstChild);
    } else {
        contentDiv.innerHTML = html;
    }
}

// ============================================
// Model Info
// ============================================

async function getModelInfo() {
    try {
        const response = await fetch('/api/model');
        if (!response.ok) throw new Error('Failed to get model info');
        
        const data = await response.json();
        modelName.textContent = data.model || 'Unknown';
    } catch (error) {
        console.error('Error getting model info:', error);
        modelName.textContent = 'Error loading';
    }
}

// ============================================
// Utilities
// ============================================

function getTheme() {
    return localStorage.getItem('theme') || 'soft-minimal';
}

function setTheme(theme) {
    localStorage.setItem('theme', theme);
    document.documentElement.setAttribute('data-theme', theme);
}

function getUserDisplayName() {
    return localStorage.getItem('userDisplayName') || 'You';
}

function setUserDisplayName(name) {
    localStorage.setItem('userDisplayName', name || 'You');
}

// ============================================
// Background Image & Blur Management
// ============================================

function setBgImage(dataUrl) {
    localStorage.setItem('bgImage', dataUrl);
    applyBgImage();
}

function getBgImage() {
    return localStorage.getItem('bgImage');
}

function clearBgImage() {
    localStorage.removeItem('bgImage');
    applyBgImage();
    updateBgPreview();
}

function applyBgImage() {
    const bgLayer = document.getElementById('backgroundLayer');
    const bgImage = getBgImage();
    if (bgImage) {
        bgLayer.style.backgroundImage = `url(${bgImage})`;
    } else {
        bgLayer.style.backgroundImage = '';
    }
}

function selectBgImage() {
    document.getElementById('bgImageInput').click();
}

function updateBgPreview() {
    const preview = document.getElementById('bgPreview');
    const bgImage = getBgImage();
    if (bgImage) {
        preview.style.backgroundImage = `url(${bgImage})`;
        preview.textContent = '';
    } else {
        preview.style.backgroundImage = '';
        preview.textContent = 'No image selected';
    }
}

function setBgBlur(value) {
    localStorage.setItem('bgBlur', value);
    applyBgBlur();
}

function getBgBlur() {
    return localStorage.getItem('bgBlur') || '30';
}

function applyBgBlur() {
    const blur = getBgBlur();
    document.documentElement.style.setProperty('--bg-blur', blur + 'px');
}

// ============================================
// UI Opacity Management
// ============================================

function setBgOpacity(value) {
    localStorage.setItem('bgOpacity', value);
    applyBgOpacity();
}

function getBgOpacity() {
    return localStorage.getItem('bgOpacity') || '80';
}

function applyBgOpacity() {
    const opacity = getBgOpacity();
    document.documentElement.style.setProperty('--bg-opacity', opacity / 100);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateString) {
    if (!dateString) return '';
    
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
        return 'Yesterday';
    } else if (diffDays < 7) {
        return date.toLocaleDateString([], { weekday: 'short' });
    } else {
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
}

function scrollToBottom(force = false) {
    // Time-based lock: scroll if lock time passed or forced
    if (force || Date.now() > scrollLockUntil) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        return marked.parse(text || '');
    }
    return text || '';
}

function renderMath() {
    if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(messagesWrapper, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\[', right: '\\]', display: true },
                { left: '\\(', right: '\\)', display: false }
            ],
            throwOnError: false
        });
    }
}

// ============================================
// Loading State
// ============================================

function showLoading(text = 'Loading...') {
    loadingOverlay.querySelector('.loading-text').textContent = text;
    loadingOverlay.classList.add('active');
}

function hideLoading() {
    loadingOverlay.classList.remove('active');
}

// ============================================
// Streaming Control
// ============================================

function setStreamingUI(streaming) {
    if (streaming) {
        sendBtn.classList.add('streaming');
        sendBtn.disabled = false;
        sendIcon.style.display = 'none';
        stopIcon.style.display = 'block';
        sendBtn.title = 'Stop generation';
    } else {
        sendBtn.classList.remove('streaming');
        sendBtn.disabled = false;
        sendIcon.style.display = 'block';
        stopIcon.style.display = 'none';
        sendBtn.title = 'Send message (Enter)';
    }
}

function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
        // Also notify server to stop
        fetch('/api/stop', { method: 'POST' }).catch(() => {});
    }
}

// ============================================
// Edit and Regenerate
// ============================================

async function editMessage(index) {
    if (!currentConversationId || isStreaming) return;
    
    const message = currentMessages[index];
    if (!message || message.role !== 'user') return;
    
    // Find message element
    const messageEl = document.querySelector(`.message[data-index="${index}"]`);
    if (!messageEl) return;
    
    // Ignore if already editing
    if (messageEl.classList.contains('editing')) return;
    
    // Enter edit mode
    editingIndex = index;
    messageEl.classList.add('editing');
    
    const contentDiv = messageEl.querySelector('.answer-content');
    
    // Save current width and set as min-width to maintain width
    const currentWidth = contentDiv.offsetWidth;
    contentDiv.style.minWidth = currentWidth + 'px';
    
    const originalContent = message.content;
    
    // Replace with textarea
    contentDiv.innerHTML = `
        <textarea class="edit-textarea">${escapeHtml(originalContent)}</textarea>
        <div class="edit-actions">
            <button class="edit-btn save" onclick="saveEdit(${index})">Save & Send</button>
            <button class="edit-btn cancel" onclick="cancelEdit(${index})">Cancel</button>
        </div>
    `;
    
    const textarea = contentDiv.querySelector('.edit-textarea');
    
    // Auto-adjust height to fit content
    function autoResize() {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }
    autoResize();
    textarea.addEventListener('input', autoResize);
    
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    
    // Save with Enter, cancel with ESC
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            saveEdit(index);
        }
        if (e.key === 'Escape') {
            cancelEdit(index);
        }
    });
}

async function saveEdit(index) {
    const messageEl = document.querySelector(`.message[data-index="${index}"]`);
    if (!messageEl) return;
    
    const textarea = messageEl.querySelector('.edit-textarea');
    if (!textarea) return;
    
    const newContent = textarea.value.trim();
    if (!newContent) return;
    
    editingIndex = -1;
    
    // Truncate and send
    try {
        const response = await fetch(`/api/conversation/${currentConversationId}/truncate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ from_index: index })
        });
        
        if (response.ok) {
            // Update UI first (reflect deletion of previous messages)
            await loadConversation(currentConversationId);
        }
        
        // Then send new message
        await sendMessage(newContent);
    } catch (error) {
        console.error('Error saving edit:', error);
    }
}

function cancelEdit(index) {
    editingIndex = -1;
    loadConversation(currentConversationId);
}

async function copyMessage(index) {
    if (!currentConversationId) return;
    
    const message = currentMessages[index];
    if (!message) return;
    
    try {
        await navigator.clipboard.writeText(message.content);
        // Simple feedback (change button icon)
        const btn = document.querySelector(`.message:nth-child(${index + 1}) .message-action-btn[onclick*="copyMessage"]`);
        if (btn) {
            const originalHTML = btn.innerHTML;
            btn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
            `;
            setTimeout(() => {
                btn.innerHTML = originalHTML;
            }, 1500);
        }
    } catch (error) {
        console.error('Failed to copy:', error);
    }
}

async function deleteFromMessage(index) {
    if (!currentConversationId) return;
    
    // Stop first if generating
    if (isStreaming) {
        stopGeneration();
        await new Promise(r => setTimeout(r, 500));  // Brief wait
    }
    
    // Confirmation dialog
    if (!confirm('Delete this message and all following messages?')) return;
    
    try {
        const response = await fetch(`/api/conversation/${currentConversationId}/truncate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ from_index: index })
        });
        
        if (response.ok) {
            await loadConversation(currentConversationId);
        }
    } catch (error) {
        console.error('Error deleting messages:', error);
    }
}

async function regenerateFrom(index) {
    if (!currentConversationId || isStreaming) return;
    
    // Get the user message before this assistant message
    const prevMessage = currentMessages[index - 1];
    if (!prevMessage || prevMessage.role !== 'user') return;
    
    const userContent = prevMessage.content;
    
    // Truncate from the user message (index - 1)
    // sendMessage will re-add the user message
    try {
        const response = await fetch(`/api/conversation/${currentConversationId}/truncate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ from_index: index - 1 })
        });
        
        if (response.ok) {
            // Reload and regenerate
            await loadConversation(currentConversationId);
            // Send the same user message again
            await sendMessage(userContent);
        }
    } catch (error) {
        console.error('Error regenerating:', error);
    }
}

// ============================================
// Modal Functions
// ============================================

function openModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

// Rename Modal
document.getElementById('renameBtn').addEventListener('click', async () => {
    if (!currentConversationId) {
        alert('No conversation selected');
        return;
    }
    
    // Get current title
    const conv = conversations.find(c => c.id === currentConversationId);
    document.getElementById('renameInput').value = conv?.title || '';
    openModal('renameModal');
    document.getElementById('renameInput').focus();
});

async function saveRename() {
    const newName = document.getElementById('renameInput').value.trim();
    if (!newName) return;
    
    try {
        const response = await fetch(`/api/conversation/${currentConversationId}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newName })
        });
        
        if (!response.ok) throw new Error('Failed to rename');
        
        await loadConversations();
        closeModal('renameModal');
    } catch (error) {
        console.error('Error renaming:', error);
        alert('Failed to rename conversation');
    }
}

// System Prompt Modal
document.getElementById('systemPromptBtn').addEventListener('click', async () => {
    try {
        const response = await fetch('/api/system_prompt');
        if (!response.ok) throw new Error('Failed to get system prompt');
        
        const data = await response.json();
        document.getElementById('systemPromptInput').value = data.system_prompt || '';
        openModal('systemPromptModal');
    } catch (error) {
        console.error('Error getting system prompt:', error);
    }
});

async function saveSystemPrompt() {
    const systemPrompt = document.getElementById('systemPromptInput').value;
    
    try {
        const response = await fetch('/api/system_prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ system_prompt: systemPrompt })
        });
        
        if (!response.ok) throw new Error('Failed to save');
        
        closeModal('systemPromptModal');
    } catch (error) {
        console.error('Error saving system prompt:', error);
        alert('Failed to save system prompt');
    }
}

// Settings Modal
document.getElementById('settingsBtn').addEventListener('click', async () => {
    try {
        const response = await fetch('/api/settings');
        if (!response.ok) throw new Error('Failed to get settings');
        
        const data = await response.json();
        // Highlight current theme button
        const currentTheme = getTheme();
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === currentTheme);
        });
        // Update background image preview
        updateBgPreview();
        // Update blur slider value
        const blur = getBgBlur();
        document.getElementById('settingBlur').value = blur;
        document.getElementById('blurValue').textContent = blur;
        // Update opacity slider value
        const opacity = getBgOpacity();
        document.getElementById('settingOpacity').value = opacity;
        document.getElementById('opacityValue').textContent = opacity;
        
        document.getElementById('settingUserName').value = getUserDisplayName();
        document.getElementById('settingTemperature').value = data.temperature || 1.0;
        document.getElementById('settingMaxLength').value = data.max_length || 32768;
        document.getElementById('settingTopK').value = data.top_k || 50;
        
        const maxAttachments = getMaxAttachments();
        document.getElementById('settingMaxAttachments').value = maxAttachments;
        document.getElementById('maxAttachmentsValue').textContent = maxAttachments;
        
        // Update Max Context value
        const maxContext = data.max_context || 32768;
        document.getElementById('settingMaxContext').value = maxContext;
        document.getElementById('maxContextWarning').style.display = 'none';
        
        // Populate language selector
        const langSelect = document.getElementById('settingLanguage');
        langSelect.innerHTML = '';
        for (const lang of getAvailableLanguages()) {
            const opt = document.createElement('option');
            opt.value = lang.code;
            opt.textContent = lang.name;
            langSelect.appendChild(opt);
        }
        langSelect.value = getCurrentLanguage();
        
        openModal('settingsModal');
    } catch (error) {
        console.error('Error getting settings:', error);
    }
});

// Theme button click event (apply and save immediately)
document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active from all buttons
        document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
        // Add active to clicked button
        btn.classList.add('active');
        // Apply theme
        setTheme(btn.dataset.theme);
    });
});

// Background image file selection event
document.getElementById('bgImageInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            setBgImage(event.target.result);
            updateBgPreview();
        };
        reader.readAsDataURL(file);
    }
});

// Blur slider event (apply in real-time)
document.getElementById('settingBlur').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('blurValue').textContent = value;
    setBgBlur(value);
});

// Opacity slider event (apply in real-time)
document.getElementById('settingOpacity').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('opacityValue').textContent = value;
    setBgOpacity(value);
});

document.getElementById('settingMaxAttachments').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('maxAttachmentsValue').textContent = value;
});

// Max Context input event (warn if exceeds 256k)
document.getElementById('settingMaxContext').addEventListener('input', (e) => {
    const value = parseInt(e.target.value);
    const warning = document.getElementById('maxContextWarning');
    if (value > 262144) {
        e.target.value = 262144;
        warning.style.display = 'block';
    } else {
        warning.style.display = 'none';
    }
});

async function saveSettings() {
    const userName = document.getElementById('settingUserName').value.trim() || 'You';
    const temperature = parseFloat(document.getElementById('settingTemperature').value);
    const maxLength = parseInt(document.getElementById('settingMaxLength').value);
    const topK = parseInt(document.getElementById('settingTopK').value);
    const maxAttachments = parseInt(document.getElementById('settingMaxAttachments').value);
    let maxContext = parseInt(document.getElementById('settingMaxContext').value);
    
    // Cap max_context at 256k
    if (maxContext > 262144) {
        maxContext = 262144;
    }
    if (maxContext < 1024) {
        maxContext = 1024;
    }
    
    setUserDisplayName(userName);
    localStorage.setItem('maxAttachments', maxAttachments.toString());
    
    // Apply language change
    const selectedLang = document.getElementById('settingLanguage').value;
    if (selectedLang !== getCurrentLanguage()) {
        await setLanguage(selectedLang);
    }
    
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ temperature, max_length: maxLength, top_k: topK, max_context: maxContext })
        });
        
        if (!response.ok) throw new Error('Failed to save');
        
        // Reload messages to update display names
        if (currentConversationId) {
            await loadConversation(currentConversationId);
        }
        
        closeModal('settingsModal');
    } catch (error) {
        console.error('Error saving settings:', error);
        alert('Failed to save settings');
    }
}

// Close modal on background click
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.active').forEach(modal => {
            modal.classList.remove('active');
        });
    }
});

// Enter key to save in rename modal
document.getElementById('renameInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        saveRename();
    }
});

// ============================================
// Tool Call / Plan Box Functions
// ============================================

let toolCallBoxes = {}; // Track tool call boxes by call_id

/**
 * Create a plan steps box (for create_plan tool calls)
 * Renders steps as numbered cards without raw JSON
 */
function createPlanStepsBox(toolCall) {
    const box = document.createElement('div');
    box.className = 'plan-steps-box';
    box.id = 'current-plan-box';
    
    const args = toolCall.arguments || {};
    const goal = args.goal || '';
    const steps = args.steps || [];
    
    // Goal header with plan reference button
    let goalHTML = goal ? `
        <div class="plan-goal plan-goal-row">
            <span class="plan-goal-text">${escapeHtml(goal)}</span>
            <button class="plan-ref-btn" onclick="askAboutPlan(this)" title="${t('tooltip.plan_ref')}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            </button>
        </div>
    ` : '';
    
    // Steps with result placeholders and toggle
    let stepsHTML = '';
    steps.forEach((step, index) => {
        const stepNum = index + 1;
        stepsHTML += `
            <div class="plan-step pending" data-step-id="${stepNum}" data-tool="${escapeHtml(step.tool || '')}">
                <div class="step-header">
                    <div class="step-header-main" onclick="toggleStepResult(this.parentElement)" ondblclick="scrollToStepOutput(${index})">
                        <div class="step-indicator">${stepNum}</div>
                        <div class="step-content">
                            <div class="step-name">${escapeHtml(step.name || '')}</div>
                            <div class="step-tool">${escapeHtml(step.tool || '')}</div>
                        </div>
                    </div>
                    <div class="step-actions">
                        <button class="step-action-btn" onclick="event.stopPropagation(); retryStep(${stepNum})" title="${t('tooltip.retry')}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M23 4v6h-6M1 20v-6h6"/>
                                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); editStepResult(${stepNum})" title="${t('tooltip.edit_result')}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); askAboutStep(${stepNum})" title="${t('tooltip.ask')}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                                <line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                        </button>
                    </div>
                    <div class="step-toggle" onclick="toggleStepResult(this.closest('.step-header'))" style="visibility: hidden;">▼</div>
                </div>
                <div class="step-result" style="display: none;"></div>
            </div>
        `;
    });
    
    box.innerHTML = `${goalHTML}<div class="plan-steps">${stepsHTML}</div>`;
    toolCallBoxes['create_plan'] = box.id;
    return box;
}

/**
 * Toggle collapse/expand of step result section
 */
function toggleStepResult(header) {
    const step = header.closest('.plan-step');
    const result = step.querySelector('.step-result');
    const toggle = step.querySelector('.step-toggle');
    
    if (result && result.innerHTML.trim()) {
        const isHidden = result.style.display === 'none';
        result.style.display = isHidden ? 'block' : 'none';
        if (toggle) {
            toggle.textContent = isHidden ? '▲' : '▼';
        }
    }
}

/**
 * Toggle collapse/expand of think section
 */
function toggleThinkSection(toggle) {
    const section = toggle.closest('.think-section-minimal');
    section.classList.toggle('collapsed');
    toggle.textContent = section.classList.contains('collapsed') ? 'Think ▶' : 'Think ▼';
}

// ============================================
// Step Action Functions (retry, edit, question)
// ============================================

// Store for step edits (save edited result - not resend!)
let stepEdits = {};

/**
 * Retry a step (LLM re-proceeds from tool select)
 */
async function retryStep(stepNum) {
    if (!currentConversationId || isStreaming) return;
    
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const tool = step.dataset.tool;
    const stepName = step.querySelector('.step-name')?.textContent || '';
    const resultEl = step.querySelector('.step-result');
    const originalResult = resultEl?.innerText || '';
    const userEdit = stepEdits[stepNum] || null;
    
    // Collect plan context
    const planBox = step.closest('.plan-steps-box');
    const planGoal = planBox?.querySelector('.plan-goal')?.textContent || '';
    
    // Collect previous steps with results
    const allSteps = planBox?.querySelectorAll('.plan-step') || [];
    const previousSteps = [];
    allSteps.forEach(s => {
        const sNum = parseInt(s.dataset.stepId);
        if (sNum < stepNum) {
            previousSteps.push({
                num: sNum,
                name: s.querySelector('.step-name')?.textContent || '',
                result: s.querySelector('.step-result')?.innerText || ''
            });
        }
    });
    
    try {
        // Show loading state
        step.classList.remove('completed', 'error');
        step.classList.add('running');
        resultEl.innerHTML = '<div class="step-loading">' + t('status.regenerating') + '</div>';
        resultEl.style.display = 'block';
        
        const response = await fetch('/retry_step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conv_id: currentConversationId,
                step_num: stepNum,
                tool: tool,
                step_name: stepName,
                original_result: originalResult,
                user_edit: userEdit,
                plan_goal: planGoal,
                previous_steps: previousSteps
            })
        });
        
        if (response.ok) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let llmText = '';
            let toolResultContent = '';
            let buffer = '';  // Buffer for incomplete SSE lines
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();  // Keep incomplete line for next chunk
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            // LLM token streaming
                            if (data.token) {
                                llmText += data.token;
                                // Show LLM thinking (optional)
                            }
                            
                            // Tool call started
                            if (data.tool_call) {
                                resultEl.innerHTML = '<div class="step-loading">' + t('status.executing', {name: data.tool_call.name}) + '</div>';
                            }
                            
                            // Tool result received
                            if (data.tool_result) {
                                toolResultContent = formatStepResult(data.tool_result.result);
                                resultEl.innerHTML = toolResultContent;
                            }
                            
                            // Legacy: direct result (for backward compatibility)
                            if (data.result) {
                                toolResultContent = formatStepResult(data.result);
                                resultEl.innerHTML = toolResultContent;
                            }
                            
                            if (data.done) {
                                step.classList.remove('running');
                                step.classList.add('completed');
                                step.querySelector('.step-indicator').textContent = '✓';
                            }
                            
                            if (data.error) {
                                step.classList.remove('running');
                                step.classList.add('error');
                                resultEl.innerHTML = `<div class="step-error">${escapeHtml(data.error)}</div>`;
                            }
                        } catch (e) {}
                    }
                }
            }
        }
    } catch (error) {
        console.error('Retry step error:', error);
        step.classList.remove('running');
        step.classList.add('error');
        resultEl.innerHTML = '<div class="step-error">' + t('error.retry_failed', {message: escapeHtml(error.message)}) + '</div>';
    }
}

/**
 * Edit step result (show edit UI - only save, not resend!)
 * Think and metadata (token/time) cannot be edited
 */
function editStepResult(stepNum) {
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const resultEl = step.querySelector('.step-result');
    if (!resultEl) return;
    
    // Check if already in edit mode
    if (step.querySelector('.step-edit-container')) return;
    
    // Get editable text only (exclude Think and metadata)
    let editableText = stepEdits[stepNum] || '';
    if (!editableText) {
        // Clone result element to manipulate
        const clone = resultEl.cloneNode(true);
        
        // Remove Think section (not editable - doesn't affect next steps)
        const thinkSection = clone.querySelector('.think-section-minimal');
        if (thinkSection) thinkSection.remove();
        
        // Remove metadata (duration, tokens - not editable)
        const metaSection = clone.querySelector('.result-meta-minimal');
        if (metaSection) metaSection.remove();
        
        // Remove result title/summary (not editable)
        const labelSection = clone.querySelector('.section-label-minimal');
        if (labelSection) labelSection.remove();
        
        editableText = clone.innerText.trim();
    }
    
    // Create edit container
    const editContainer = document.createElement('div');
    editContainer.className = 'step-edit-container';
    editContainer.innerHTML = `
        <textarea class="step-edit-textarea" placeholder="${t('placeholder.edit_result')}">${escapeHtml(editableText)}</textarea>
        <div class="step-edit-actions">
            <button class="step-edit-btn cancel" onclick="cancelStepEdit(${stepNum})">${t('label.cancel')}</button>
            <button class="step-edit-btn save" onclick="saveStepEdit(${stepNum})">${t('label.save')}</button>
        </div>
    `;
    
    // Insert after result
    resultEl.style.display = 'none';
    step.appendChild(editContainer);
    
    // Focus textarea
    const textarea = editContainer.querySelector('textarea');
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

/**
 * Save step edit (save edited content - do not resend!)
 */
function saveStepEdit(stepNum) {
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const textarea = step.querySelector('.step-edit-textarea');
    const editContainer = step.querySelector('.step-edit-container');
    const resultEl = step.querySelector('.step-result');
    
    if (textarea && textarea.value.trim()) {
        // Save to stepEdits (save to memory, not resend)
        stepEdits[stepNum] = textarea.value.trim();
        
        // Add edited badge to step name if not exists
        const stepName = step.querySelector('.step-name');
        if (stepName && !stepName.querySelector('.step-edited-badge')) {
            const badge = document.createElement('span');
            badge.className = 'step-edited-badge';
            badge.textContent = t('label.modified');
            stepName.appendChild(badge);
        }
    }
    
    // Remove edit container and show result
    if (editContainer) editContainer.remove();
    if (resultEl) resultEl.style.display = 'block';
}

/**
 * Cancel step edit
 */
function cancelStepEdit(stepNum) {
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const editContainer = step.querySelector('.step-edit-container');
    const resultEl = step.querySelector('.step-result');
    
    if (editContainer) editContainer.remove();
    if (resultEl) resultEl.style.display = 'block';
}

/**
 * Ask about entire plan (ask referencing the full Plan)
 */
function askAboutPlan(btn) {
    const planBox = btn.closest('.plan-steps-box');
    const planGoal = planBox?.querySelector('.plan-goal')?.textContent || '';
    
    // Collect all steps with results
    const allSteps = planBox?.querySelectorAll('.plan-step') || [];
    const planSteps = [];
    allSteps.forEach(s => {
        planSteps.push({
            num: parseInt(s.dataset.stepId),
            name: s.querySelector('.step-name')?.textContent || '',
            tool: s.dataset.tool || '',
            result: s.querySelector('.step-result')?.innerText || ''
        });
    });
    
    // Set context for plan-level question
    currentStepQuestion = { 
        stepNum: 0,  // 0 = plan level
        tool: 'plan',
        stepName: t('label.entire_plan'),
        context: '',
        previousSteps: [],
        planGoal,
        planSteps
    };
    
    // Add tag pill
    const inputTags = document.getElementById('inputTags');
    inputTags.innerHTML = `
        <span class="input-tag" data-step="plan">
            ${t('label.entire_plan')}
            <span class="input-tag-remove" onclick="removeStepTag()">×</span>
        </span>
    `;
    
    messageInput.value = '';
    messageInput.placeholder = t('placeholder.plan_question');
    messageInput.focus();
    messageInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Ask about step (add tag to main input field)
 */
function askAboutStep(stepNum) {
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const tool = step.dataset.tool || '';
    const stepName = step.querySelector('.step-name')?.textContent || '';
    const resultEl = step.querySelector('.step-result');
    const context = resultEl?.innerText || '';
    
    // Collect plan info (goal + all steps structure)
    const planBox = step.closest('.plan-steps-box');
    const planGoal = planBox?.querySelector('.plan-goal')?.textContent || '';
    
    // Collect all steps structure (for plan context)
    const allStepsInPlan = planBox?.querySelectorAll('.plan-step') || [];
    const planSteps = [];
    allStepsInPlan.forEach(s => {
        const sNum = parseInt(s.dataset.stepId);
        planSteps.push({
            num: sNum,
            name: s.querySelector('.step-name')?.textContent || '',
            tool: s.dataset.tool || ''
        });
    });
    
    // Collect previous steps with results (for detailed context)
    const previousSteps = [];
    allStepsInPlan.forEach(s => {
        const sNum = parseInt(s.dataset.stepId);
        if (sNum < stepNum) {
            previousSteps.push({
                num: sNum,
                name: s.querySelector('.step-name')?.textContent || '',
                result: s.querySelector('.step-result')?.innerText || ''
            });
        }
    });
    
    // Store context for when message is sent
    currentStepQuestion = { 
        stepNum, tool, stepName, context, 
        previousSteps,
        planGoal,      // Plan goal
        planSteps      // Full plan structure (for reference when creating new plan)
    };
    
    // Add tag pill (instead of text)
    const inputTags = document.getElementById('inputTags');
    inputTags.innerHTML = `
        <span class="input-tag" data-step="${stepNum}">
            Step ${stepNum}
            <span class="input-tag-remove" onclick="removeStepTag()">×</span>
        </span>
    `;
    
    // Clear input and focus
    messageInput.value = '';
    messageInput.placeholder = t('placeholder.step_question', {num: stepNum});
    messageInput.focus();
    
    // Scroll to input area
    messageInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Remove step tag from input
 */
function removeStepTag() {
    const inputTags = document.getElementById('inputTags');
    inputTags.innerHTML = '';
    currentStepQuestion = null;
    messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
}

/**
 * Send question about step to LLM
 */
async function sendStepQuestion(stepNum) {
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const input = step.querySelector('.step-question-input');
    const question = input?.value.trim();
    if (!question) return;
    
    const tool = step.dataset.tool;
    const stepName = step.querySelector('.step-name')?.textContent || '';
    const resultEl = step.querySelector('.step-result');
    const stepContext = resultEl?.innerText || '';
    
    // Disable input
    input.disabled = true;
    const sendBtn = step.querySelector('.step-question-send');
    if (sendBtn) sendBtn.disabled = true;
    
    // Remove existing answer
    const existingAnswer = step.querySelector('.step-question-answer');
    if (existingAnswer) existingAnswer.remove();
    
    // Create answer container
    const answerDiv = document.createElement('div');
    answerDiv.className = 'step-question-answer';
    answerDiv.innerHTML = '<span class="loading-dots">' + t('status.generating_answer') + '</span>';
    step.appendChild(answerDiv);
    
    try {
        const response = await fetch('/step_question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conv_id: currentConversationId,
                step_num: stepNum,
                tool: tool,
                step_name: stepName,
                step_context: stepContext,
                question: question
            })
        });
        
        if (response.ok) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let answerText = '';
            let buffer = '';  // Buffer for incomplete SSE lines
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();  // Keep incomplete line for next chunk
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.token) {
                                answerText += data.token;
                                answerDiv.innerHTML = renderMarkdown(answerText);
                            }
                            if (data.done) {
                                // Clear input
                                input.value = '';
                            }
                        } catch (e) {}
                    }
                }
            }
        }
    } catch (error) {
        console.error('Step question error:', error);
        answerDiv.innerHTML = '<span class="error">' + t('error.generic', {message: escapeHtml(error.message)}) + '</span>';
    } finally {
        input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        input.focus();
    }
}

/**
 * Send step question from main input (question with @StepN: tag)
 */
async function sendStepQuestionFromMain(question) {
    if (!currentStepQuestion || !currentConversationId) {
        // Remove tag and return
        document.getElementById('inputTags').innerHTML = '';
        messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
        currentStepQuestion = null;
        return;
    }
    
    const { stepNum, tool, stepName, context, previousSteps, planGoal, planSteps } = currentStepQuestion;
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Remove tag immediately (at send time)
    document.getElementById('inputTags').innerHTML = '';
    messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
    
    // Hide welcome message
    if (welcomeMessage) welcomeMessage.style.display = 'none';
    
    // Show user's question in chat (using createMessageHTML)
    // Use marker that will be converted to tag pill after rendering
    const userMsgContent = `{{STEP_TAG:${stepNum}}} ${question}`;
    
    // Add to currentMessages for proper indexing
    currentMessages.push({role: 'user', content: userMsgContent});
    const userIndex = currentMessages.length - 1;
    
    const userHTML = createMessageHTML({role: 'user', content: userMsgContent}, userIndex);
    messagesWrapper.insertAdjacentHTML('beforeend', userHTML);
    scrollToBottom();
    
    // Create assistant message placeholder (using createMessageHTML)
    // Add placeholder to currentMessages
    currentMessages.push({role: 'assistant', content: ''});
    const assistantIndex = currentMessages.length - 1;
    
    const assistantHTML = createMessageHTML({role: 'assistant', content: ''}, assistantIndex);
    messagesWrapper.insertAdjacentHTML('beforeend', assistantHTML);
    const assistantMsgEl = messagesWrapper.lastElementChild;
    const contentDiv = assistantMsgEl.querySelector('.answer-content');
    
    // Safety check
    if (!contentDiv) {
        console.error('contentDiv not found in assistant message');
        // Remove tag and return
        document.getElementById('inputTags').innerHTML = '';
        messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
        currentStepQuestion = null;
        return;
    }
    
    // Show streaming indicator
    contentDiv.innerHTML = '<span class="streaming-indicator"><span class="streaming-dot"></span><span class="streaming-dot"></span><span class="streaming-dot"></span></span>';
    
    isStreaming = true;
    setStreamingUI(true);
    
    try {
        const response = await fetch('/step_question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conv_id: currentConversationId,
                step_num: stepNum,
                tool: tool,
                step_name: stepName,
                step_context: context,
                previous_steps: previousSteps,
                plan_goal: planGoal,
                plan_steps: planSteps,
                question: question
            })
        });
        
        console.log('Step question response:', response.status, response.ok);
        
        if (response.ok) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let answerText = '';
            let buffer = '';  // Buffer for incomplete SSE lines
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();  // Keep incomplete line for next chunk
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.token) {
                                answerText += data.token;
                                contentDiv.innerHTML = renderMarkdown(answerText);
                                scrollToBottom();
                            }
                            if (data.error) {
                                contentDiv.innerHTML = '<span class="error">' + t('error.generic', {message: escapeHtml(data.error)}) + '</span>';
                            }
                        } catch (e) {}
                    }
                }
            }
            
            // Final render
            if (answerText) {
                contentDiv.innerHTML = renderMarkdown(answerText);
                // Update currentMessages with final answer
                currentMessages[assistantIndex] = {role: 'assistant', content: answerText};
            }
            
            // Reload sidebar to show updated conversation
            loadConversations();
        } else {
            contentDiv.innerHTML = '<span class="error">' + t('error.request_failed') + '</span>';
        }
    } catch (error) {
        console.error('Step question error:', error);
        contentDiv.innerHTML = '<span class="error">' + t('error.generic', {message: escapeHtml(error.message)}) + '</span>';
    } finally {
        isStreaming = false;
        setStreamingUI(false);
        currentStepQuestion = null;
        // Ensure tag is removed
        document.getElementById('inputTags').innerHTML = '';
        messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
    }
}

/**
 * Merge multiple tool results for the same step into a per-step map.
 * Backend stores all_results as a flat array where the same step number
 * can appear multiple times when a step triggers several tool calls.
 */
function mergeResultsByStep(results) {
    const map = {};
    for (const r of results) {
        if (!map[r.step]) {
            map[r.step] = { ...r, _merged: [r] };
        } else {
            const m = map[r.step];
            m._merged.push(r);
            if (r.success === false) m.success = false;
            if (r.stopped) m.stopped = true;
        }
    }
    return map;
}

/**
 * Format tool result for display in a plan step
 * Minimal text-only design without backgrounds, borders, or icons
 */
function formatStepResult(toolResult) {
    if (!toolResult) return '';

    let html = '';

    if (typeof toolResult === 'string') {
        return `<div class="section-text">${renderMarkdown(toolResult)}</div>`;
    }

    if (toolResult.thought) {
        html += `
            <div class="think-section-minimal collapsed">
                <span class="think-toggle" onclick="toggleThinkSection(this)">Think ▶</span>
                <div class="think-content">${renderMarkdown(toolResult.thought)}</div>
            </div>
        `;
    }

    if (toolResult.error) {
        html += `<div class="step-error">${escapeHtml(toolResult.error)}</div>`;
    }

    const result = toolResult.result || toolResult;

    // If the result contains code (code_gen), show summary and execution results
    if (result && result.code) {
        const lineCount = result.code.split('\n').length;
        const execId = 'exec-' + Math.random().toString(36).slice(2, 9);
        let fixInfo = '';
        if (result.fix_attempts > 0) {
            fixInfo = `<div class="code-fix-info">Auto-corrected (${result.fix_attempts} ${result.fix_attempts === 1 ? 'retry' : 'retries'})</div>`;
        }
        html += `<div class="step-section-minimal">
            <span class="section-label-minimal">Code Generated</span>
            <div class="code-gen-summary">${lineCount} lines of ${escapeHtml(result.language || 'python')} (view in Code tab)</div>
            ${fixInfo}
            <div class="step-exec-result" id="${execId}"></div>
        </div>`;
        if (result.execution) {
            setTimeout(() => {
                const container = document.getElementById(execId);
                if (container) container.innerHTML = renderExecutionResult(result.execution);
            }, 50);
        } else if (toolResult.step) {
            setTimeout(() => autoExecuteCode(toolResult.step - 1, result.code, result.language || 'python', execId), 100);
        }
        return html;
    }

    if (result && (result.title || result.details)) {
        const detailsHtml = result.details && Array.isArray(result.details)
            ? `<ul class="result-details-minimal">${result.details.map(d => `<li>${renderMarkdown(d)}</li>`).join('')}</ul>`
            : '';

        let graphHtml = '';
        if (result.has_graph) {
            if (result.graph_type === 'efficiency') {
                graphHtml = createEfficiencyChart(result);
            } else if (result.graph_type === 'timeline') {
                graphHtml = createTimelineChart(result);
            }
        }

        const metaHtml = (result.duration || result.tokens)
            ? `<div class="result-meta-minimal">${result.duration || ''}${result.duration && result.tokens ? ' · ' : ''}${result.tokens ? result.tokens + ' tokens' : ''}</div>`
            : '';

        html += `
            <div class="step-section-minimal">
                <span class="section-label-minimal">${escapeHtml(result.title || t('label.result'))}</span>
                ${detailsHtml}
                ${graphHtml}
                ${metaHtml}
            </div>
        `;
    } else if (!html && result) {
        html = `<pre class="result-json">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
    }

    return html;
}

function renderExecutionResult(execution) {
    if (!execution) return '';
    let html = '';
    if (execution.stdout && execution.stdout.trim()) {
        html += `<pre class="code-stdout">${escapeHtml(execution.stdout)}</pre>`;
    }
    if (execution.figures && execution.figures.length) {
        html += execution.figures.map(f => `<img src="${f}" class="code-result-img">`).join('');
    }
    if (execution.tables && execution.tables.length) {
        const tblId = 'tbl-' + Math.random().toString(36).slice(2, 9);
        html += `<div id="${tblId}" class="code-result-tables"></div>`;
        setTimeout(() => {
            const tblContainer = document.getElementById(tblId);
            if (tblContainer) {
                execution.tables.forEach(csvUrl => loadCsvTable(csvUrl, tblContainer));
            }
        }, 50);
    }
    if (execution.stderr && execution.stderr.trim()) {
        html += `<pre class="code-error">${escapeHtml(execution.stderr)}</pre>`;
    }
    return html;
}

async function autoExecuteCode(stepIndex, code, language, execContainerId) {
    if (!currentConversationId) return;
    try {
        const res = await fetch('/api/execute_code', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, language, conv_id: currentConversationId, step_index: stepIndex })
        });
        const data = await res.json();
        const container = document.getElementById(execContainerId);
        if (container) {
            let html = '';
            if (data.stdout && data.stdout.trim()) {
                html += `<pre class="code-stdout">${escapeHtml(data.stdout)}</pre>`;
            }
            if (data.figures && data.figures.length) {
                html += data.figures.map(f => `<img src="${f}" class="code-result-img">`).join('');
            }
            if (data.tables && data.tables.length) {
                data.tables.forEach(csvUrl => {
                    loadCsvTable(csvUrl, container);
                });
            }
            if (data.stderr && data.stderr.trim()) {
                html += `<pre class="code-error">${escapeHtml(data.stderr)}</pre>`;
            }
            container.innerHTML = html;
        }
        // Also refresh code tab if visible
        const codeContent = document.getElementById('codeContent');
        if (codeContent) {
            const block = codeContent.querySelector(`.code-block[data-step="${stepIndex}"]`);
            if (block) {
                const resultDiv = block.querySelector('.code-result');
                if (resultDiv) displayCodeResult(resultDiv, data);
            }
        }
    } catch (e) {
        console.warn('Auto-execute code failed:', e);
    }
}

async function loadSavedOutputsForPlanBox() {
    if (!currentConversationId) return;
    const planBox = document.querySelector('.completed-plan');
    if (!planBox) return;
    const stepEls = planBox.querySelectorAll('.plan-step[data-step-id]');
    for (const stepEl of stepEls) {
        const stepNum = parseInt(stepEl.dataset.stepId);
        const stepIndex = stepNum - 1;
        try {
            const res = await fetch(`/api/outputs/${currentConversationId}/step_${stepIndex}`);
            const data = await res.json();
            if ((!data.figures || !data.figures.length) && (!data.tables || !data.tables.length)) continue;
            let resultEl = stepEl.querySelector('.step-result');
            if (!resultEl) {
                resultEl = document.createElement('div');
                resultEl.className = 'step-result';
                resultEl.style.display = 'block';
                stepEl.appendChild(resultEl);
            }
            let execDiv = resultEl.querySelector('.step-exec-result');
            if (!execDiv) {
                execDiv = document.createElement('div');
                execDiv.className = 'step-exec-result';
                resultEl.appendChild(execDiv);
            }
            let html = '';
            if (data.figures && data.figures.length) {
                html += data.figures.map(f => `<img src="${f}" class="code-result-img">`).join('');
            }
            execDiv.innerHTML = html;
            if (data.tables && data.tables.length) {
                for (const csvUrl of data.tables) {
                    await loadCsvTable(csvUrl, execDiv);
                }
            }
        } catch (e) {
            // No saved outputs
        }
    }
}

/**
 * Update a plan step with its tool result
 * Returns true if the result was successfully associated with a plan step
 */
function updatePlanStepResult(toolName, toolResult) {
    const planBox = document.getElementById('current-plan-box');
    if (!planBox) return false;
    
    // First try matching by step number (more reliable)
    let stepEl = null;
    if (toolResult.step) {
        stepEl = planBox.querySelector(`[data-step-id="${toolResult.step}"]`);
    }
    // Fallback: find the step with matching tool name - prefer running step first
    if (!stepEl) {
        stepEl = planBox.querySelector(`.plan-step.running[data-tool="${toolName}"]`) ||
                 planBox.querySelector(`.plan-step.pending[data-tool="${toolName}"]`) ||
                 planBox.querySelector(`[data-tool="${toolName}"]`);
    }
    if (!stepEl) return false;
    
    // Update step status
    stepEl.classList.remove('pending', 'running');
    stepEl.classList.add(toolResult.success ? 'completed' : 'error');
    
    // Update indicator
    const indicator = stepEl.querySelector('.step-indicator');
    if (indicator) {
        indicator.textContent = toolResult.success ? '✓' : '!';
    }
    
    // Update tool name display (tool is selected at execution time)
    const toolEl = stepEl.querySelector('.step-tool');
    if (toolEl && toolName) {
        toolEl.textContent = toolName;
        stepEl.dataset.tool = toolName;
    }
    
    // Show result - append if results already exist (multi-tool step)
    const resultEl = stepEl.querySelector('.step-result');
    if (resultEl && toolResult) {
        resultEl.style.display = 'block';
        const newHtml = formatStepResult(toolResult);
        if (resultEl.innerHTML.trim()) {
            resultEl.innerHTML += '<hr class="tool-result-divider">' + newHtml;
        } else {
            resultEl.innerHTML = newHtml;
        }

        if (typeof renderMathInElement !== 'undefined') {
            renderMathInElement(resultEl, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '\\(', right: '\\)', display: false }
                ],
                throwOnError: false
            });
        }

        // Update toggle arrow to indicate content is available
        const toggle = stepEl.querySelector('.step-toggle');
        if (toggle) {
            toggle.style.visibility = 'visible';
            toggle.textContent = '▲';
        }
    }
    
    return true;
}

/**
 * Update plan box with completed results from plan_complete data
 * Called when done event includes plan_complete
 */
function updatePlanBoxWithResults(planBox, planData) {
    const mergedMap = mergeResultsByStep(planData.results || []);
    
    for (const [step, merged] of Object.entries(mergedMap)) {
        const stepEl = planBox.querySelector(`[data-step-id="${step}"]`);
        if (!stepEl) continue;
        
        stepEl.classList.remove('pending', 'running');
        const stepStatus = merged.stopped ? 'stopped' : (merged.success ? 'completed' : 'error');
        stepEl.classList.add(stepStatus);
        
        const indicator = stepEl.querySelector('.step-indicator');
        if (indicator) {
            indicator.textContent = merged.stopped ? '◼' : (merged.success ? '✓' : '!');
        }
        
        const resultEl = stepEl.querySelector('.step-result');
        const items = merged._merged || [merged];
        const hasContent = items.some(r => r.thought || r.action || r.result || r.error);
        if (resultEl && hasContent) {
            resultEl.style.display = 'block';
            resultEl.innerHTML = items.map(r => formatStepResult({
                success: r.success,
                thought: r.thought,
                action: r.action,
                error: r.error,
                result: r.result
            })).join('<hr class="tool-result-divider">');
            
            const toggle = stepEl.querySelector('.step-toggle');
            if (toggle) {
                toggle.style.visibility = 'visible';
                toggle.textContent = '▲';
            }
        }
    }
    
    // Set remaining steps to "stopped" if plan was stopped
    if (planData.stopped) {
        planBox.querySelectorAll('.plan-step.pending').forEach(stepEl => {
            stepEl.classList.remove('pending');
            stepEl.classList.add('stopped');
            const indicator = stepEl.querySelector('.step-indicator');
            if (indicator) indicator.textContent = '◼';
        });
    }
}

/**
 * Create a tool call box (shown when LLM calls a tool)
 */
function createToolCallBox(toolCall) {
    const box = document.createElement('div');
    box.className = 'tool-call-box';
    box.dataset.toolName = toolCall.name;
    box.dataset.status = toolCall.status || 'running';
    
    const toolId = `tool-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    box.id = toolId;
    toolCallBoxes[toolCall.name] = toolId;
    
    // Tool icon mapping
    const toolIcons = {
        'pubmed_search': '📚',
        'ncbi_gene': '🧬',
        'crispr_designer': '✂️',
        'protocol_builder': '📋',
        'create_plan': '📝',
        'execute_step': '▶️'
    };
    
    const icon = toolIcons[toolCall.name] || '🔧';
    
    box.innerHTML = `
        <div class="tool-call-header">
            <span class="tool-icon">${icon}</span>
            <span class="tool-name">${escapeHtml(toolCall.name)}</span>
            <span class="tool-status ${toolCall.status || 'running'}">${toolCall.status === 'running' ? t('status.running') : t('status.completed')}</span>
        </div>
        <div class="tool-call-args">
            <pre>${escapeHtml(JSON.stringify(toolCall.arguments, null, 2))}</pre>
        </div>
        <div class="tool-call-result" style="display: none;"></div>
    `;
    
    return box;
}

/**
 * Update a tool result box with the result data
 */
function updateToolResultBox(toolResult) {
    const toolName = toolResult.tool || toolResult.name || toolResult.executed_tool;
    
    // First, try to update plan step if this tool is part of a plan
    if (updatePlanStepResult(toolName, toolResult)) {
        return; // Result shown in plan step
    }
    
    // Fallback: show in separate tool box
    const boxId = toolCallBoxes[toolName];
    
    if (!boxId) return;
    
    const box = document.getElementById(boxId);
    if (!box) return;
    
    // Update status
    box.dataset.status = toolResult.success ? 'completed' : 'error';
    const statusEl = box.querySelector('.tool-status');
    if (statusEl) {
        statusEl.className = `tool-status ${toolResult.success ? 'completed' : 'error'}`;
        statusEl.textContent = toolResult.success ? t('status.completed') : t('status.error');
    }
    
    // Show result
    const resultEl = box.querySelector('.tool-call-result');
    if (resultEl && toolResult.result) {
        resultEl.style.display = 'block';
        
        const result = toolResult.result;
        let resultHTML = '';
        
        // Add thought if present
        if (toolResult.thought) {
            resultHTML += `<div class="tool-thought">💭 ${escapeHtml(toolResult.thought)}</div>`;
        }
        
        // Add action if present
        if (toolResult.action) {
            resultHTML += `<div class="tool-action">🔧 ${escapeHtml(toolResult.action)}</div>`;
        }
        
        // Add result content
        if (result.title) {
            resultHTML += `<div class="tool-result-title">✅ ${escapeHtml(result.title)}</div>`;
        }
        
        if (result.details && Array.isArray(result.details)) {
            resultHTML += '<ul class="tool-result-details">';
            for (const detail of result.details) {
                resultHTML += `<li>${escapeHtml(detail)}</li>`;
            }
            resultHTML += '</ul>';
        }
        
        // Add duration/tokens info if present
        if (result.duration || result.tokens) {
            resultHTML += `<div class="tool-meta">`;
            if (result.duration) resultHTML += `⏱️ ${result.duration} `;
            if (result.tokens) resultHTML += `📊 ${result.tokens} tokens`;
            resultHTML += `</div>`;
        }
        
        resultEl.innerHTML = resultHTML;
    }
}

/**
 * Create minimal efficiency bar chart (sgRNA efficiency distribution)
 */
function createEfficiencyChart(result) {
    // Generate bar data based on avg_efficiency or random
    const avgEff = result.avg_efficiency || 0.72;
    const bars = [];
    for (let i = 0; i < 6; i++) {
        const value = 0.4 + (i * 0.1);
        // Distribution: more bars near avg efficiency
        const height = Math.max(20, Math.min(95, 
            100 - Math.abs(value - avgEff) * 200 + Math.random() * 30
        ));
        bars.push({ value: value.toFixed(1), height: Math.round(height) });
    }
    
    const barsHtml = bars.map(b => 
        `<div class="mini-bar" style="height: ${b.height}%;" title="${b.value}"></div>`
    ).join('');
    
    const labelsHtml = bars.map(b => 
        `<span class="mini-bar-label">${b.value}</span>`
    ).join('');
    
    return `
        <div class="mini-graph-container">
            <div class="mini-graph-title">${t('label.sgrna_distribution')}</div>
            <div class="mini-chart">
                ${barsHtml}
            </div>
            <div class="mini-chart-labels">
                ${labelsHtml}
            </div>
        </div>
    `;
}

/**
 * Create minimal timeline chart (experiment schedule)
 */
function createTimelineChart(result) {
    const weeks = result.duration_weeks || 6;
    const phases = [];
    
    // Determine phases based on experiment type
    if (result.experiment_type === 'crispr_screen' || !result.experiment_type) {
        const p1 = Math.ceil(weeks / 3);
        const p2 = Math.ceil(2 * weeks / 3);
        phases.push({ weeks: p1, label: t('label.cloning'), color: '#8B4513' });
        phases.push({ weeks: p2 - p1, label: t('label.transduction'), color: '#4A5568' });
        phases.push({ weeks: weeks - p2, label: t('label.analysis'), color: '#22C55E' });
    } else {
        // Generic phases
        const half = Math.ceil(weeks / 2);
        phases.push({ weeks: half, label: t('label.preparation'), color: '#8B4513' });
        phases.push({ weeks: weeks - half, label: t('label.execution'), color: '#22C55E' });
    }
    
    // Build week boxes
    let weekBoxes = '';
    let weekNum = 1;
    let labelHtml = '';
    
    phases.forEach((phase, pi) => {
        for (let i = 0; i < phase.weeks && weekNum <= weeks; i++) {
            weekBoxes += `<div class="timeline-week" style="background: ${phase.color};">W${weekNum}</div>`;
            weekNum++;
        }
        labelHtml += `<span class="timeline-phase-label" style="flex: ${phase.weeks};">${phase.label}</span>`;
    });
    
    return `
        <div class="mini-graph-container">
            <div class="mini-graph-title">${t('label.week_timeline', {weeks})}</div>
            <div class="mini-timeline">
                ${weekBoxes}
            </div>
            <div class="mini-timeline-labels">
                ${labelHtml}
            </div>
        </div>
    `;
}

/**
 * Create a plan box showing all steps
 */
function createPlanBox(plan) {
    const box = document.createElement('div');
    box.className = 'plan-box';
    box.id = 'current-plan-box';
    
    let stepsHTML = '';
    for (const step of plan.steps || []) {
        const statusClass = step.status || 'pending';
        const statusIcon = statusClass === 'completed' ? '✓' : statusClass === 'running' ? '···' : step.id;
        
        stepsHTML += `
            <div class="plan-step ${statusClass}" data-step-id="${step.id}">
                <div class="step-indicator">${statusIcon}</div>
                <div class="step-content">
                    <div class="step-name">${escapeHtml(step.name)}</div>
                    <div class="step-tool">${escapeHtml(step.tool)}</div>
                </div>
            </div>
        `;
    }
    
    box.innerHTML = `
        <div class="plan-header">
            <span class="plan-icon">📋</span>
            <span class="plan-title">${t('label.execution_plan')}</span>
            <span class="plan-progress">${plan.current_step || 0} / ${plan.total_steps || plan.steps?.length || 0}</span>
        </div>
        <div class="plan-goal">${escapeHtml(plan.goal || '')}</div>
        <div class="plan-steps">${stepsHTML}</div>
    `;
    
    return box;
}

/**
 * Update plan step status
 */
function updatePlanStep(stepId, status) {
    const planBox = document.getElementById('current-plan-box');
    if (!planBox) return;
    
    const stepEl = planBox.querySelector(`[data-step-id="${stepId}"]`);
    if (!stepEl) return;
    
    stepEl.className = `plan-step ${status}`;
    const indicator = stepEl.querySelector('.step-indicator');
    if (indicator) {
        indicator.textContent = status === 'completed' ? '✓' : status === 'running' ? '···' : stepId;
    }
}

// ============================================
// Detail Panel Functions
// ============================================

/**
 * Initialize Detail Panel event listeners
 */
function setupDetailPanelListeners() {
    // Close button
    if (detailClose) {
        detailClose.addEventListener('click', closeDetailPanel);
    }
    
    // Toggle button
    if (detailToggle) {
        detailToggle.addEventListener('click', toggleDetailPanel);
    }
    
    // Tab switching
    document.querySelectorAll('.detail-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            switchDetailTab(tab.dataset.tab);
        });
    });
    
    // Resize handle
    if (detailResizeHandle) {
        setupDetailResize();
    }
    
    // Regenerate buttons
    const regeneratePlan = document.getElementById('regeneratePlan');
    if (regeneratePlan) {
        regeneratePlan.addEventListener('click', () => requestAnalyzePlan(true));
    }
    
    const regenerateCode = document.getElementById('regenerateCode');
    if (regenerateCode) {
        regenerateCode.addEventListener('click', () => {
            if (currentCodeStep !== null) {
                requestCodeGen(currentCodeStep, true);
            }
        });
    }
    
    // Copy code button
    const copyCodeBtn = document.getElementById('copyCodeBtn');
    if (copyCodeBtn) {
        copyCodeBtn.addEventListener('click', copyCurrentCode);
    }
}

/**
 * Open Detail Panel with plan data
 */
function openDetailPanel(planData) {
    if (!detailPanel) return;
    
    // Initialize data
    detailPanelData.goal = planData.goal || '';
    detailPanelData.userMessage = planData.userMessage || '';
    detailPanelData.steps = planData.steps || [];
    detailPanelData.results = [];
    detailPanelData.codes = {};
    detailPanelData.analysis = '';
    detailPanelData.currentStep = 0;
    currentCodeStep = null;
    
    // Ensure panel is visible
    showDetailPanelUI();
    
    // Hide empty states, show content
    const planEmptyState = document.getElementById('planEmptyState');
    if (planEmptyState) planEmptyState.style.display = 'none';
    const graphEmptyState = document.getElementById('graphEmptyState');
    if (graphEmptyState) graphEmptyState.style.display = 'none';
    const nodeGraphContainer = document.getElementById('nodeGraphContainer');
    if (nodeGraphContainer) nodeGraphContainer.style.display = '';
    
    // Show "executing plan" loading state (analysis will be generated after all steps complete)
    const planLoading = document.getElementById('planLoading');
    if (planLoading) {
        planLoading.style.display = 'flex';
        const loadingText = planLoading.querySelector('.loading-text');
        if (loadingText) loadingText.textContent = t('status.executing_plan');
    }
    
    // Create graph from plan data
    if (nodeGraph) {
        nodeGraph.createFromPlan(planData);
        const graphTabVisible = document.getElementById('tabGraph')?.style.display !== 'none';
        if (graphTabVisible) {
            setTimeout(() => {
                nodeGraph._relayoutVertical();
                nodeGraph.fitToView();
                if (currentConversationId) {
                    localStorage.setItem('graphState-' + currentConversationId, JSON.stringify(nodeGraph.getState()));
                }
                nodeGraph._needsLayout = null;
            }, 100);
        } else {
            nodeGraph._needsLayout = 'full';
        }
        // Notify popout if open
        broadcastGraphMessage({ type: 'plan-data', payload: planData });
    }
}

function showDetailPanelUI() {
    if (detailPanel) detailPanel.style.display = 'flex';
    document.documentElement.style.setProperty('--detail-panel-width', detailPanelWidth + 'px');
    if (detailResizeHandle) detailResizeHandle.style.display = 'block';
    if (detailToggle) {
        detailToggle.style.display = 'flex';
        detailToggle.classList.add('panel-open');
    }
    detailPanelOpen = true;
}

/**
 * Close Detail Panel (hide, but can be toggled back)
 */
function closeDetailPanel() {
    if (!detailPanel) return;
    
    detailPanel.style.display = 'none';
    if (detailResizeHandle) detailResizeHandle.style.display = 'none';
    if (detailToggle) detailToggle.classList.remove('panel-open');
    detailPanelOpen = false;
}

/**
 * Toggle Detail Panel
 */
function toggleDetailPanel() {
    if (detailPanelOpen) {
        closeDetailPanel();
    } else {
        showDetailPanelUI();
    }
}

/**
 * Reset Detail Panel to empty state (keep panel visible)
 */
function hideDetailPanel() {
    // Reset detail panel data
    detailPanelData = {
        goal: '',
        steps: [],
        results: {},
        codes: {},
        analysis: null,
        currentStep: 0
    };
    
    // Show empty states
    const planEmptyState = document.getElementById('planEmptyState');
    if (planEmptyState) planEmptyState.style.display = 'flex';
    const graphEmptyState = document.getElementById('graphEmptyState');
    if (graphEmptyState) graphEmptyState.style.display = 'flex';
    const nodeGraphContainer = document.getElementById('nodeGraphContainer');
    if (nodeGraphContainer) nodeGraphContainer.style.display = 'none';
    
    // Hide loading / content
    const planLoading = document.getElementById('planLoading');
    if (planLoading) planLoading.style.display = 'none';
    const planContent = document.getElementById('planContent');
    if (planContent) planContent.innerHTML = '';
    const regeneratePlan = document.getElementById('regeneratePlan');
    if (regeneratePlan) regeneratePlan.style.display = 'none';
    
    // Clear graph
    if (nodeGraph) nodeGraph.clear();
}

/**
 * Restore detail panel from saved conversation messages.
 * Scans for the last [PLAN_COMPLETE] message and rebuilds the panel.
 */
function restoreDetailPanelFromMessages(messages) {
    let lastPlanData = null;
    let lastPlanMsgIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg.role === 'assistant' && msg.content) {
            const match = msg.content.match(/\[PLAN_COMPLETE\]([\s\S]*)$/);
            if (match) {
                try {
                    lastPlanData = JSON.parse(match[1].trim());
                    lastPlanMsgIdx = i;
                } catch (e) { /* ignore parse errors */ }
                break;
            }
        }
    }
    
    if (lastPlanData) {
        for (let j = lastPlanMsgIdx - 1; j >= 0; j--) {
            if (messages[j].role === 'user') {
                lastPlanData.userMessage = (messages[j].content || '').replace(/\[Image: [^\]]+\]\s*/g, '').replace(/\[Audio: [^\]]+\]\s*/g, '').replace(/\[Document: [^\]]+\]\s*/g, '').trim();
                break;
            }
        }
        openDetailPanelFromSaved(lastPlanData);
        return;
    }
    
    let toolCallPlan = null;
    let userGoal = '';
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg.role === 'assistant' && msg.content) {
            const parsed = parseSpecialTokens(msg.content);
            if (parsed.toolCalls && typeof parsed.toolCalls === 'object' && parsed.toolCalls.name === 'create_plan') {
                toolCallPlan = parsed.toolCalls.arguments;
                for (let j = i - 1; j >= 0; j--) {
                    if (messages[j].role === 'user') {
                        userGoal = (messages[j].content || '').replace(/\[Image: [^\]]+\]\s*/g, '').replace(/\[Audio: [^\]]+\]\s*/g, '').replace(/\[Document: [^\]]+\]\s*/g, '').trim();
                        break;
                    }
                }
                break;
            }
        }
    }
    
    if (toolCallPlan) {
        openDetailPanel({
            goal: toolCallPlan.goal || userGoal || '',
            userMessage: userGoal || '',
            steps: (toolCallPlan.steps || []).map((s, i) => ({
                id: i + 1,
                name: s.name || '',
                tool: s.tool || '',
                description: s.description || ''
            }))
        });
    } else {
        hideDetailPanel();
    }
}

/**
 * Open detail panel from saved plan data (including results).
 * Similar to openDetailPanel but restores completed results too.
 */
function openDetailPanelFromSaved(planData) {
    if (!detailPanel) return;
    
    detailPanelData.goal = planData.goal || '';
    detailPanelData.userMessage = planData.userMessage || '';
    detailPanelData.steps = planData.steps || [];
    
    // Restore results indexed by step number (1-based -> 0-based)
    const results = planData.results || [];
    detailPanelData.results = [];
    results.forEach(r => {
        if (r.step) detailPanelData.results[r.step - 1] = r;
    });
    
    detailPanelData.codes = {};
    results.forEach(r => {
        if (r.step && r.result?.code) {
            detailPanelData.codes[r.step - 1] = {
                language: r.result.language || 'python',
                code: r.result.code
            };
        }
    });
    currentCodeStep = null;
    detailPanelData.analysis = planData.analysis || '';
    detailPanelData.currentStep = detailPanelData.steps.length;
    
    showDetailPanelUI();
    
    // Hide empty states, show content
    const planEmptyState = document.getElementById('planEmptyState');
    if (planEmptyState) planEmptyState.style.display = 'none';
    const graphEmptyState = document.getElementById('graphEmptyState');
    if (graphEmptyState) graphEmptyState.style.display = 'none';
    const nodeGraphContainer = document.getElementById('nodeGraphContainer');
    if (nodeGraphContainer) nodeGraphContainer.style.display = '';
    
    // Restore graph: prefer saved full state, fall back to createFromPlan
    if (nodeGraph) {
        let restoredFromState = false;
        if (currentConversationId) {
            const savedState = localStorage.getItem('graphState-' + currentConversationId);
            if (savedState) {
                try {
                    nodeGraph.setState(JSON.parse(savedState));
                    restoredFromState = true;
                } catch (e) {}
            }
        }
        if (!restoredFromState) {
            nodeGraph.createFromPlan(planData);
            const graphTabVisible = document.getElementById('tabGraph')?.style.display !== 'none';
            if (graphTabVisible) {
                setTimeout(() => {
                    nodeGraph._relayoutVertical();
                    nodeGraph.fitToView();
                    if (currentConversationId) {
                        localStorage.setItem('graphState-' + currentConversationId, JSON.stringify(nodeGraph.getState()));
                    }
                    nodeGraph._needsLayout = null;
                }, 100);
            } else {
                nodeGraph._needsLayout = 'full';
            }
        }
        // Apply step statuses from results (may have updated since last save)
        results.forEach(r => {
            if (r.step) {
                const nodeId = `step-${r.step}`;
                const status = r.stopped ? 'stopped' : (r.success ? 'completed' : 'error');
                nodeGraph.setNodeStatus(nodeId, status);
            }
        });
        if (planData.stopped) {
            (planData.steps || []).forEach((step, index) => {
                const stepNum = index + 1;
                const hasResult = results.some(r => r.step === stepNum);
                if (!hasResult) {
                    nodeGraph.setNodeStatus(`step-${stepNum}`, 'stopped');
                }
            });
        }
        broadcastGraphMessage({ type: 'plan-data', payload: planData });
    }
    
    // Restore saved analysis or request fresh one
    if (detailPanelData.analysis) {
        renderAnalysisPlan(detailPanelData.analysis);
        const regenerateBtn = document.getElementById('regeneratePlan');
        if (regenerateBtn) regenerateBtn.style.display = 'inline-flex';
    } else {
        const regenerateBtn = document.getElementById('regeneratePlan');
        if (regenerateBtn) regenerateBtn.style.display = 'inline-flex';
    }
    renderOutputs();
    updateCodeStepSelector();
    setTimeout(() => loadSavedOutputsForPlanBox(), 200);
}

// ============================================
// Mode Toggle
// ============================================

function applyModeToggle() {
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === currentMode);
    });
}

// ============================================
// Graph Popout & BroadcastChannel
// ============================================

let graphChannel = null;

function setupGraphChannel() {
    if (typeof BroadcastChannel === 'undefined') return;
    graphChannel = new BroadcastChannel('node-graph');
    graphChannel.onmessage = (e) => {
        const { type, payload } = e.data;
        switch (type) {
            case 'popout-closed':
                onGraphPopoutClosed();
                break;
            case 'graph-modified':
                if (nodeGraph) nodeGraph.setState(payload);
                break;
            case 'rerun-plan':
                rerunPlanFromGraph();
                break;
            case 'request-state':
                broadcastGraphMessage({
                    type: 'plan-data',
                    payload: {
                        goal: detailPanelData.goal,
                        steps: detailPanelData.steps
                    }
                });
                broadcastGraphMessage({
                    type: 'graph-state',
                    payload: nodeGraph ? nodeGraph.getState() : null
                });
                broadcastGraphMessage({
                    type: 'theme-change',
                    payload: getTheme()
                });
                break;
        }
    };
}

function broadcastGraphMessage(msg) {
    if (graphChannel) graphChannel.postMessage(msg);
}

function openGraphPopout() {
    if (graphPopoutWindow && !graphPopoutWindow.closed) {
        graphPopoutWindow.focus();
        return;
    }
    graphPopoutWindow = window.open('/graph.html', 'graphEditor', 'width=1200,height=800');
    
    // Show notice in detail panel, hide graph container
    const notice = document.getElementById('graphPopoutNotice');
    const container = document.getElementById('nodeGraphContainer');
    if (notice) notice.style.display = 'flex';
    if (container) container.style.display = 'none';
}

function onGraphPopoutClosed() {
    graphPopoutWindow = null;
    const notice = document.getElementById('graphPopoutNotice');
    const container = document.getElementById('nodeGraphContainer');
    if (notice) notice.style.display = 'none';
    if (container) container.style.display = '';
}

async function rerunPlanFromGraph() {
    if (!nodeGraph || !currentConversationId || isStreaming) return;

    const executionPlan = nodeGraph.toExecutionPlan();
    if (!executionPlan || !executionPlan.steps || executionPlan.steps.length === 0) return;

    const logicChanged = nodeGraph.hasExecutionLogicChanged();

    if (logicChanged) {
        const oldPlanBox = document.getElementById('current-plan-box');
        if (oldPlanBox) {
            oldPlanBox.removeAttribute('id');
            oldPlanBox.classList.add('plan-box-archived');
        }

        const toolCall = {
            name: 'create_plan',
            arguments: { goal: executionPlan.goal, steps: executionPlan.steps }
        };
        const newPlanBox = createPlanStepsBox(toolCall);

        const assistantMsg = appendMessage({ role: 'assistant', content: '' });
        const contentDiv = assistantMsg.querySelector('.message-content');
        contentDiv.innerHTML = '';
        contentDiv.appendChild(newPlanBox);

        const planSteps = executionPlan.steps.map((s, i) => ({
            id: i + 1,
            name: s.name || '',
            tool: s.tool || '',
            description: s.description || ''
        }));
        openDetailPanel({ goal: executionPlan.goal, userMessage: detailPanelData.userMessage || '', steps: planSteps });
    } else {
        const planBox = document.getElementById('current-plan-box');
        if (planBox) {
            planBox.querySelectorAll('.plan-step').forEach(step => {
                step.classList.remove('completed', 'running', 'error');
                step.classList.add('pending');
                const indicator = step.querySelector('.step-indicator');
                if (indicator) indicator.textContent = step.dataset.stepId || '';
                const result = step.querySelector('.step-result');
                if (result) { result.style.display = 'none'; result.innerHTML = ''; }
                const toggle = step.querySelector('.step-toggle');
                if (toggle) toggle.style.visibility = 'hidden';
            });
        }
    }

    try {
        const response = await fetch('/api/replan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conversation_id: currentConversationId,
                goal: executionPlan.goal,
                steps: executionPlan.steps,
                rerun: true
            })
        });
        if (!response.ok) {
            console.error('Replan failed:', response.statusText);
            return;
        }

        isStreaming = true;
        currentAbortController = new AbortController();
        setStreamingUI(true);

        const chatResponse = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conversation_id: currentConversationId,
                message: `[RERUN_PLAN] Execute the updated plan.`,
                mode: currentMode
            }),
            signal: currentAbortController.signal
        });

        if (!chatResponse.ok) throw new Error('Rerun chat failed');

        const reader = chatResponse.body.getReader();
        const decoder = new TextDecoder();
        let fullContent = '';
        let buffer = '';

        const planBox = document.getElementById('current-plan-box');
        const contentDiv = planBox?.closest('.message-content');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));

                    if (data.tool_call && data.tool_call.name !== 'create_plan') {
                        if (planBox) {
                            const toolName = data.tool_call.name;
                            const stepEl = planBox.querySelector(`.plan-step.pending[data-tool="${toolName}"]`) ||
                                           planBox.querySelector(`.plan-step:not(.completed):not(.error)[data-tool="${toolName}"]`);
                            if (stepEl && !stepEl.classList.contains('completed')) {
                                stepEl.classList.remove('pending');
                                stepEl.classList.add('running');
                                void stepEl.offsetHeight;
                            }
                        }
                    }

                    if (data.tool_result) {
                        updateToolResultBox(data.tool_result);
                        if (detailPanelOpen && data.tool_result.step !== undefined) {
                            addToolResultToDetailPanel(data.tool_result.step - 1, data.tool_result);
                        }
                        if (nodeGraph && data.tool_result.step !== undefined) {
                            const status = data.tool_result.success ? 'completed' : 'error';
                            nodeGraph.setNodeStatus(`step-${data.tool_result.step}`, status);
                            broadcastGraphMessage({ type: 'step-update', payload: { step: data.tool_result.step, status } });
                        }
                    }

                    if (data.step_start) {
                        if (planBox) {
                            const stepEl = planBox.querySelector(`[data-step-id="${data.step_start.step}"]`);
                            if (stepEl && !stepEl.classList.contains('completed')) {
                                stepEl.classList.remove('pending');
                                stepEl.classList.add('running');
                            }
                        }
                        if (nodeGraph) {
                            nodeGraph.setNodeStatus(`step-${data.step_start.step}`, 'running');
                        }
                        broadcastGraphMessage({ type: 'step-update', payload: { step: data.step_start.step, status: 'running' } });
                    }

                    if (data.done) {
                        if (data.plan_complete && planBox) {
                            updatePlanBoxWithResults(planBox, data.plan_complete);
                        }
                        if (detailPanelOpen && data.plan_complete) {
                            onPlanComplete();
                        }
                    }
                } catch (e) {
                    console.warn('SSE parse error during rerun:', e);
                }
            }
        }
    } catch (err) {
        if (err.name !== 'AbortError') {
            console.error('Rerun error:', err);
        }
    } finally {
        isStreaming = false;
        currentAbortController = null;
        setStreamingUI(false);
        nodeGraph?.snapshotExecutionPlanHash();
    }
}

/**
 * Switch Detail Panel tabs
 */
function switchDetailTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.detail-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    // Update tab content
    document.querySelectorAll('.detail-tab-content').forEach(content => {
        content.style.display = 'none';
        content.classList.remove('active');
    });
    
    const activeContent = document.getElementById('tab' + tabName.charAt(0).toUpperCase() + tabName.slice(1));
    if (activeContent) {
        activeContent.style.display = 'flex';
        activeContent.classList.add('active');
    }

    if (tabName === 'graph' && typeof nodeGraph !== 'undefined' && nodeGraph) {
        setTimeout(() => {
            if (nodeGraph._needsLayout) {
                nodeGraph._relayoutVertical();
                if (currentConversationId) {
                    localStorage.setItem('graphState-' + currentConversationId, JSON.stringify(nodeGraph.getState()));
                }
            }
            nodeGraph.fitToView();
            nodeGraph._needsLayout = null;
        }, 100);
    }
}

/**
 * Setup resize drag functionality
 */
function setupDetailResize() {
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;
    
    detailResizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startWidth = detailPanel.offsetWidth;
        detailResizeHandle.classList.add('resizing');
        // Add resizing class to toggle to disable transition
        if (detailToggle) {
            detailToggle.classList.add('resizing');
        }
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        
        const diff = startX - e.clientX;
        const containerWidth = document.querySelector('.chat-detail-container').clientWidth;
        const newWidth = Math.max(250, Math.min(startWidth + diff, containerWidth * 0.6));
        
        detailPanelWidth = newWidth;
        // Update CSS variable on :root for consistent usage
        document.documentElement.style.setProperty('--detail-panel-width', newWidth + 'px');
        detailPanel.style.width = newWidth + 'px';
        
        // Sync toggle button position (no transition during resize)
        if (detailToggle && detailPanelOpen) {
            detailToggle.style.right = (newWidth + 4) + 'px';
        }
    });
    
    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            detailResizeHandle.classList.remove('resizing');
            // Remove resizing class from toggle
            if (detailToggle) {
                detailToggle.classList.remove('resizing');
                // Clear inline style, let CSS handle via variable
                detailToggle.style.right = '';
            }
            // Clear inline width, let CSS handle via variable
            detailPanel.style.width = '';
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

/**
 * Request analyze_plan tool call
 */
async function requestAnalyzePlan(force = false) {
    const planLoading = document.getElementById('planLoading');
    const planContent = document.getElementById('planContent');
    const regenerateBtn = document.getElementById('regeneratePlan');
    
    if (!force && detailPanelData.analysis) {
        // Already have analysis
        return;
    }
    
    // Show loading
    if (planLoading) {
        planLoading.style.display = 'flex';
        const loadingText = planLoading.querySelector('.loading-text');
        if (loadingText) loadingText.textContent = t('status.analyzing');
    }
    if (planContent) planContent.innerHTML = '';
    if (regenerateBtn) regenerateBtn.style.display = 'none';
    
    try {
        // Build steps with results
        const stepsWithResults = detailPanelData.steps.map((step, i) => ({
            name: step.name,
            tool: step.tool,
            description: step.description || '',
            status: i < detailPanelData.currentStep ? 'completed' : 
                   i === detailPanelData.currentStep ? 'running' : 'pending',
            result: detailPanelData.results[i] || null
        }));
        
        const response = await fetch('/api/tool_call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tool: 'analyze_plan',
                args: {
                    goal: detailPanelData.goal,
                    steps: stepsWithResults,
                    current_step: detailPanelData.currentStep
                }
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.result && data.result.analysis) {
            detailPanelData.analysis = data.result.analysis;
            renderAnalysisPlan(data.result.analysis);
            saveAnalysisToConversation(data.result.analysis);
        } else {
            if (planContent) {
                planContent.innerHTML = '<div class="error">' + t('error.analysis_failed') + '</div>';
            }
        }
    } catch (error) {
        console.error('analyze_plan error:', error);
        if (planContent) {
            planContent.innerHTML = '<div class="error">' + t('error.generic', {message: escapeHtml(error.message)}) + '</div>';
        }
    } finally {
        if (planLoading) planLoading.style.display = 'none';
        if (regenerateBtn) regenerateBtn.style.display = 'inline-flex';
    }
}

/**
 * Render Analysis Plan content (markdown)
 */
function renderAnalysisPlan(analysis) {
    const planContent = document.getElementById('planContent');
    if (!planContent) return;
    
    const html = renderMarkdown(analysis);
    planContent.innerHTML = html;

    if (html === analysis) {
        console.warn('renderAnalysisPlan: marked may not be loaded, markdown was not converted');
    }

    if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(planContent, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\[', right: '\\]', display: true },
                { left: '\\(', right: '\\)', display: false }
            ],
            throwOnError: false
        });
    }
}

/**
 * Persist analysis text back into the PLAN_COMPLETE message on the server.
 */
async function saveAnalysisToConversation(analysisText) {
    if (!currentConversationId) return;
    try {
        await fetch('/api/update_plan_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conversation_id: currentConversationId,
                analysis: analysisText
            })
        });
    } catch (e) {
        console.error('Failed to save analysis:', e);
    }
}

/**
 * Request code_gen tool call for a step
 */
async function requestCodeGen(stepIndex, force = false) {
    const codeLoading = document.getElementById('codeLoading');
    const codeContent = document.getElementById('codeContent');
    const codeActions = document.getElementById('codeActions');
    
    // Check if already have code for this step
    if (!force && detailPanelData.codes[stepIndex]) {
        renderCode(stepIndex);
        return;
    }
    
    const stepResult = detailPanelData.results[stepIndex];
    if (!stepResult) {
        if (codeContent) {
            codeContent.innerHTML = '<div class="code-empty-state">' + t('empty.step_no_result') + '</div>';
        }
        return;
    }
    
    // Show loading
    if (codeLoading) codeLoading.style.display = 'flex';
    if (codeActions) codeActions.style.display = 'none';
    
    const step = detailPanelData.steps[stepIndex];
    const toolName = step?.tool || 'unknown';
    
    try {
        const response = await fetch('/api/tool_call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tool: 'code_gen',
                args: {
                    task: `Generate Python code to visualize the results from ${toolName}. Use matplotlib and seaborn for plots, and pandas for data tables.`,
                    language: 'python',
                    context: (() => { const raw = JSON.stringify(stepResult.result || stepResult, null, 2); return raw.length > 2000 ? raw.substring(0, 2000) + '\n...(truncated)' : raw; })()
                }
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.result) {
            detailPanelData.codes[stepIndex] = data.result;
            renderCode(stepIndex);
        } else {
            const errMsg = data.error || data.result?.error || 'Code generation failed';
            console.error('code_gen failed:', errMsg);
            if (codeContent) {
                codeContent.innerHTML = `<div class="error">Code generation failed: ${escapeHtml(errMsg)}</div>`;
            }
        }
    } catch (error) {
        console.error('code_gen error:', error);
        if (codeContent) {
            codeContent.innerHTML = `<div class="error">Error: ${escapeHtml(error.message)}</div>`;
        }
    } finally {
        if (codeLoading) codeLoading.style.display = 'none';
    }
}

/**
 * Update code step selector buttons
 */
function updateCodeStepSelector() {
    const selector = document.getElementById('codeStepSelector');
    if (!selector) return;

    const hasAnyCode = Object.keys(detailPanelData.codes).length > 0;
    if (!hasAnyCode) {
        selector.innerHTML = '';
        renderCode('all');
        return;
    }

    if (currentCodeStep === null) currentCodeStep = 'all';

    let html = '';
    const allActive = currentCodeStep === 'all' ? ' active' : '';
    html += `<button class="code-step-btn${allActive}" data-step="all">All</button>`;

    for (let i = 0; i < detailPanelData.steps.length; i++) {
        if (!detailPanelData.codes[i]) continue;
        const step = detailPanelData.steps[i];
        const activeClass = currentCodeStep === i ? ' active' : '';
        html += `<button class="code-step-btn${activeClass}" data-step="${i}">
            Step ${i + 1}: ${escapeHtml(step?.tool || '')}
        </button>`;
    }

    selector.innerHTML = html;

    selector.querySelectorAll('.code-step-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const val = btn.dataset.step;
            currentCodeStep = val === 'all' ? 'all' : parseInt(val);
            updateCodeStepSelector();
            renderCode(currentCodeStep);
        });
    });

    renderCode(currentCodeStep);
}

/**
 * Render code for a step or summary
 */
function renderCode(stepIndex) {
    const codeContent = document.getElementById('codeContent');
    const codeActions = document.getElementById('codeActions');
    if (!codeContent) return;

    if (stepIndex === 'all') {
        let html = '';
        let hasAny = false;
        for (let i = 0; i < detailPanelData.steps.length; i++) {
            const cd = detailPanelData.codes[i];
            if (!cd || !cd.code) continue;
            hasAny = true;
            const step = detailPanelData.steps[i];
            const title = 'Step ' + (i + 1) + ': ' + (step?.tool || '');
            html += createCodeBlockHTML(cd.code, cd.language, title, i);
        }
        if (!hasAny) {
            codeContent.innerHTML = '<div class="code-empty-state">' + t('empty.no_code') + '</div>';
            if (codeActions) codeActions.style.display = 'none';
        } else {
            codeContent.innerHTML = html;
            attachCodeBlockListeners(codeContent);
            if (codeActions) codeActions.style.display = 'flex';
            loadAllSavedResults(codeContent);
        }
        currentCodeStep = 'all';
        return;
    }

    const codeData = detailPanelData.codes[stepIndex];
    const step = detailPanelData.steps[stepIndex];
    const title = 'Step ' + (stepIndex + 1) + ': ' + (step?.tool || '');

    if (!codeData || !codeData.code) {
        codeContent.innerHTML = '<div class="code-empty-state">' + t('empty.no_code') + '</div>';
        if (codeActions) codeActions.style.display = 'none';
        return;
    }

    codeContent.innerHTML = createCodeBlockHTML(codeData.code, codeData.language, title, stepIndex);
    attachCodeBlockListeners(codeContent);
    if (codeActions) codeActions.style.display = 'flex';
    currentCodeStep = stepIndex;
    loadAllSavedResults(codeContent);
}

/**
 * Syntax highlighting for Python and R code.
 */
function highlightCodeSyntax(code, language) {
    let s = escapeHtml(code);
    language = (language || 'python').toLowerCase();

    const pythonKeywords = [
        'import', 'from', 'as', 'def', 'class', 'return', 'if', 'else', 'elif',
        'for', 'while', 'in', 'not', 'and', 'or', 'True', 'False', 'None',
        'try', 'except', 'finally', 'with', 'lambda', 'yield', 'pass', 'break', 'continue'
    ];

    const rKeywords = [
        'function', 'library', 'require', 'if', 'else', 'for', 'while', 'repeat',
        'return', 'next', 'break', 'in', 'TRUE', 'FALSE', 'NULL', 'NA', 'NA_integer_',
        'NA_real_', 'NA_complex_', 'NA_character_', 'Inf', 'NaN',
        'source', 'list', 'c', 'data\\.frame', 'matrix'
    ];

    const keywords = language === 'r' ? rKeywords : pythonKeywords;

    // Placeholder tokens to avoid regex interference between passes
    const PH = '\x00';
    const CM_S = PH + 'CS' + PH, CM_E = PH + 'CE' + PH;
    const ST_S = PH + 'SS' + PH, ST_E = PH + 'SE' + PH;
    const KW_S = PH + 'KS' + PH, KW_E = PH + 'KE' + PH;
    const NM_S = PH + 'NS' + PH, NM_E = PH + 'NE' + PH;
    const FN_S = PH + 'FS' + PH, FN_E = PH + 'FE' + PH;

    s = s.replace(/(#.*)$/gm, CM_S + '$1' + CM_E);
    s = s.replace(/(&quot;.*?&quot;|&#39;.*?&#39;|"[^"]*"|'[^']*')/g, ST_S + '$1' + ST_E);
    s = s.replace(/\b(\d+\.?\d*)\b/g, NM_S + '$1' + NM_E);

    if (language === 'r') {
        s = s.replace(/(&lt;-|-&gt;|&lt;&lt;-)/g, KW_S + '$1' + KW_E);
        s = s.replace(/(%[^%]+%)/g, KW_S + '$1' + KW_E);
    }

    keywords.forEach(kw => {
        const regex = new RegExp(`\\b(${kw})\\b`, 'g');
        s = s.replace(regex, KW_S + '$1' + KW_E);
    });

    s = s.replace(/\b([a-zA-Z_][\w.]*)\s*\(/g, FN_S + '$1' + FN_E + '(');

    // Final pass: replace placeholders with actual span tags
    s = s.replace(/\x00CS\x00/g, '<span class="code-comment">');
    s = s.replace(/\x00CE\x00/g, '</span>');
    s = s.replace(/\x00SS\x00/g, '<span class="code-string">');
    s = s.replace(/\x00SE\x00/g, '</span>');
    s = s.replace(/\x00KS\x00/g, '<span class="code-keyword">');
    s = s.replace(/\x00KE\x00/g, '</span>');
    s = s.replace(/\x00NS\x00/g, '<span class="code-number">');
    s = s.replace(/\x00NE\x00/g, '</span>');
    s = s.replace(/\x00FS\x00/g, '<span class="code-function">');
    s = s.replace(/\x00FE\x00/g, '</span>');

    return s;
}

/**
 * Build a code block HTML with header, syntax highlighting, copy/run buttons, and result area.
 */
function createCodeBlockHTML(code, language, title, stepIndex) {
    const highlighted = highlightCodeSyntax(code, language);
    const id = 'cb-' + Math.random().toString(36).slice(2, 9);
    const stepAttr = stepIndex != null ? ` data-step="${stepIndex}"` : '';
    return `
        <div class="code-block" id="${id}"${stepAttr}>
            <div class="code-block-header">
                <span class="code-block-title">${escapeHtml(title || '')}</span>
                <span class="code-block-lang">${escapeHtml(language || 'python')}</span>
                <button class="code-run-btn" data-target="${id}">Run</button>
                <button class="code-copy-btn" data-target="${id}">Copy</button>
            </div>
            <div class="code-block-body">${highlighted}</div>
            <div class="code-result" id="${id}-result" style="display:none">
                <div class="code-result-stdout"></div>
                <div class="code-result-figures"></div>
                <div class="code-result-tables"></div>
                <div class="code-result-stderr"></div>
            </div>
        </div>`;
}

/**
 * Copy code from a specific code block via its copy button.
 */
function copyCodeBlock(btn) {
    const targetId = btn.dataset.target;
    const block = document.getElementById(targetId);
    if (!block) return;
    const body = block.querySelector('.code-block-body');
    if (!body) return;
    const text = body.textContent || '';
    navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = 'Copy';
            btn.classList.remove('copied');
        }, 2000);
    }).catch(err => console.error('Copy failed:', err));
}

/**
 * Attach click listeners to all .code-copy-btn elements inside a container.
 */
function attachCodeCopyListeners(container) {
    if (!container) return;
    container.querySelectorAll('.code-copy-btn').forEach(btn => {
        btn.addEventListener('click', () => copyCodeBlock(btn));
    });
}

function attachCodeBlockListeners(container) {
    if (!container) return;
    container.querySelectorAll('.code-copy-btn').forEach(btn => {
        btn.addEventListener('click', () => copyCodeBlock(btn));
    });
    container.querySelectorAll('.code-run-btn').forEach(btn => {
        btn.addEventListener('click', () => runCodeBlock(btn));
    });
}

async function runCodeBlock(btn) {
    const targetId = btn.dataset.target;
    const block = document.getElementById(targetId);
    if (!block) return;
    const body = block.querySelector('.code-block-body');
    if (!body) return;
    const code = body.textContent || '';
    const lang = block.querySelector('.code-block-lang')?.textContent || 'python';
    const stepIndex = block.dataset.step != null ? parseInt(block.dataset.step) : 0;

    btn.disabled = true;
    btn.textContent = 'Running...';

    const resultDiv = block.querySelector('.code-result');
    if (resultDiv) {
        resultDiv.style.display = 'block';
        resultDiv.querySelector('.code-result-stdout').innerHTML = '<div class="code-running">Executing...</div>';
        resultDiv.querySelector('.code-result-figures').innerHTML = '';
        resultDiv.querySelector('.code-result-tables').innerHTML = '';
        resultDiv.querySelector('.code-result-stderr').innerHTML = '';
    }

    try {
        const res = await fetch('/api/execute_code', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, language: lang, conv_id: currentConversationId, step_index: stepIndex })
        });
        const data = await res.json();
        if (resultDiv) {
            displayCodeResult(resultDiv, data);
        }
    } catch (e) {
        if (resultDiv) {
            resultDiv.querySelector('.code-result-stderr').innerHTML = `<pre class="code-error">${escapeHtml(e.message)}</pre>`;
        }
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run';
    }
}

function displayCodeResult(resultDiv, data) {
    const stdoutEl = resultDiv.querySelector('.code-result-stdout');
    const figEl = resultDiv.querySelector('.code-result-figures');
    const tblEl = resultDiv.querySelector('.code-result-tables');
    const stderrEl = resultDiv.querySelector('.code-result-stderr');

    stdoutEl.innerHTML = '';
    figEl.innerHTML = '';
    tblEl.innerHTML = '';
    stderrEl.innerHTML = '';

    let hasContent = false;

    if (data.stdout && data.stdout.trim()) {
        stdoutEl.innerHTML = `<pre class="code-stdout">${escapeHtml(data.stdout)}</pre>`;
        hasContent = true;
    }
    if (data.figures && data.figures.length) {
        figEl.innerHTML = data.figures.map(f => `<img src="${f}" class="code-result-img">`).join('');
        hasContent = true;
    }
    if (data.tables && data.tables.length) {
        data.tables.forEach(csvUrl => loadCsvTable(csvUrl, tblEl));
        hasContent = true;
    }
    if (data.stderr && data.stderr.trim()) {
        stderrEl.innerHTML = `<pre class="code-error">${escapeHtml(data.stderr)}</pre>`;
        hasContent = true;
    }

    resultDiv.style.display = hasContent ? 'block' : 'none';
}

async function loadCsvTable(csvUrl, container) {
    try {
        const res = await fetch(csvUrl);
        const text = await res.text();
        const rows = text.split('\n').filter(r => r.trim());
        if (!rows.length) return;
        const headers = parseCSVRow(rows[0]);
        let html = '<table class="csv-table"><thead><tr>';
        headers.forEach(h => { html += `<th>${escapeHtml(h)}</th>`; });
        html += '</tr></thead><tbody>';
        for (let i = 1; i < rows.length; i++) {
            const cols = parseCSVRow(rows[i]);
            html += '<tr>';
            cols.forEach(c => { html += `<td>${escapeHtml(c)}</td>`; });
            html += '</tr>';
        }
        html += '</tbody></table>';
        container.insertAdjacentHTML('beforeend', html);
    } catch (e) {
        console.warn('Failed to load CSV table:', e);
    }
}

function parseCSVRow(row) {
    const result = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < row.length; i++) {
        const ch = row[i];
        if (inQuotes) {
            if (ch === '"' && row[i + 1] === '"') {
                current += '"';
                i++;
            } else if (ch === '"') {
                inQuotes = false;
            } else {
                current += ch;
            }
        } else {
            if (ch === '"') {
                inQuotes = true;
            } else if (ch === ',') {
                result.push(current);
                current = '';
            } else {
                current += ch;
            }
        }
    }
    result.push(current);
    return result;
}

async function loadAllSavedResults(container) {
    if (!currentConversationId) return;
    const blocks = container.querySelectorAll('.code-block[data-step]');
    for (const block of blocks) {
        const stepIndex = parseInt(block.dataset.step);
        const resultDiv = block.querySelector('.code-result');
        if (!resultDiv) continue;
        try {
            const res = await fetch(`/api/outputs/${currentConversationId}/step_${stepIndex}`);
            const data = await res.json();
            let hasContent = false;
            if (data.figures && data.figures.length) {
                const figEl = resultDiv.querySelector('.code-result-figures');
                figEl.innerHTML = data.figures.map(f => `<img src="${f}" class="code-result-img">`).join('');
                hasContent = true;
            }
            if (data.tables && data.tables.length) {
                const tblEl = resultDiv.querySelector('.code-result-tables');
                for (const csvUrl of data.tables) {
                    await loadCsvTable(csvUrl, tblEl);
                }
                hasContent = true;
            }
            if (hasContent) resultDiv.style.display = 'block';
        } catch (e) {
            // No saved results for this step
        }
    }
}

/**
 * Copy current code to clipboard
 */
async function copyCurrentCode() {
    const copyBtn = document.getElementById('copyCodeBtn');

    let textToCopy = '';
    if (currentCodeStep === 'all') {
        const parts = [];
        for (let i = 0; i < detailPanelData.steps.length; i++) {
            const cd = detailPanelData.codes[i];
            if (!cd || !cd.code) continue;
            const step = detailPanelData.steps[i];
            parts.push(`# Step ${i + 1}: ${step?.tool || ''}\n${cd.code}`);
        }
        textToCopy = parts.join('\n\n');
    } else if (currentCodeStep !== null) {
        const codeData = detailPanelData.codes[currentCodeStep];
        if (codeData) textToCopy = codeData.code;
    }

    if (!textToCopy) return;
    
    try {
        await navigator.clipboard.writeText(textToCopy);
        
        if (copyBtn) {
            copyBtn.classList.add('copied');
            copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"/>
            </svg> Copied!`;
            
            setTimeout(() => {
                copyBtn.classList.remove('copied');
                copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg> Copy`;
            }, 2000);
        }
    } catch (err) {
        console.error('Copy failed:', err);
    }
}

/**
 * Render Outputs tab content
 */
function renderOutputs() {
    const outputsContent = document.getElementById('outputsContent');
    if (!outputsContent) return;
    
    if (detailPanelData.results.length === 0) {
        outputsContent.innerHTML = '<div class="outputs-empty-state">' + t('empty.outputs_hint') + '</div>';
        return;
    }
    
    let html = '';
    
    for (let i = 0; i < detailPanelData.results.length; i++) {
        const result = detailPanelData.results[i];
        const step = detailPanelData.steps[i];
        const toolResult = result.result || result;
        
        html += `
            <div class="output-step-section" id="step-output-${i}">
                <div class="output-step-header">
                    <span class="output-step-number">${i + 1}</span>
                    <span class="output-step-title">${escapeHtml(step?.name || 'Step')}</span>
                    <span class="output-step-tool">${escapeHtml(step?.tool || '')}</span>
                </div>
                <div class="output-content">
                    ${renderToolResultDetail(toolResult)}
                </div>
            </div>
        `;
    }
    
    outputsContent.innerHTML = html;
}

/**
 * Render detailed tool result for Outputs tab
 */
function renderToolResultDetail(result) {
    if (!result) return '<div class="error">' + t('error.no_result') + '</div>';
    
    let html = '';
    
    // Title
    if (result.title) {
        html += `<div class="output-title">${escapeHtml(result.title)}</div>`;
    }
    
    // Details list
    if (result.details && Array.isArray(result.details)) {
        html += '<ul class="output-details">';
        result.details.forEach(detail => {
            html += `<li>${escapeHtml(String(detail))}</li>`;
        });
        html += '</ul>';
    }
    
    // Table data (gene_table, paper_list, efficiency_data)
    if (result.gene_table && Array.isArray(result.gene_table)) {
        html += renderOutputTable(['Gene', 'Function', 'Location'], result.gene_table, ['gene', 'function', 'location']);
    }
    
    if (result.paper_list && Array.isArray(result.paper_list)) {
        html += renderOutputTable(['Title', 'Authors', 'Year'], result.paper_list.slice(0, 10), ['title', 'authors', 'year']);
    }
    
    if (result.efficiency_data && Array.isArray(result.efficiency_data)) {
        html += renderOutputTable(['Gene', 'Score'], result.efficiency_data.slice(0, 10), ['gene', 'score']);
    }
    
    // Summary
    if (result.summary) {
        html += `<div class="output-summary">${escapeHtml(result.summary)}</div>`;
    }
    
    // Meta info (duration, tokens)
    const meta = [];
    if (result.duration) meta.push(`${result.duration}s`);
    if (result.tokens) meta.push(`${result.tokens} tokens`);
    
    if (meta.length > 0) {
        html += `<div class="output-meta">${meta.join(' | ')}</div>`;
    }
    
    // Fallback to JSON if nothing else
    if (!html) {
        html = `<pre class="result-json">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
    }
    
    return html;
}

/**
 * Render table for output data
 */
function renderOutputTable(headers, data, keys) {
    if (!data || data.length === 0) return '';
    
    let html = '<table class="output-table"><thead><tr>';
    headers.forEach(h => {
        html += `<th>${escapeHtml(h)}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    data.forEach(row => {
        html += '<tr>';
        keys.forEach(key => {
            html += `<td>${escapeHtml(String(row[key] || ''))}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    return html;
}

/**
 * Add tool result to Detail Panel and trigger updates
 */
function addToolResultToDetailPanel(stepIndex, result) {
    if (!detailPanelData.results[stepIndex]) {
        detailPanelData.results[stepIndex] = result;
    } else {
        if (!Array.isArray(detailPanelData.results[stepIndex])) {
            detailPanelData.results[stepIndex] = [detailPanelData.results[stepIndex]];
        }
        detailPanelData.results[stepIndex].push(result);
    }
    detailPanelData.currentStep = stepIndex + 1;

    if (result.result?.code) {
        detailPanelData.codes[stepIndex] = {
            language: result.result.language || 'python',
            code: result.result.code
        };
    }

    renderOutputs();
    updateCodeStepSelector();
}

/**
 * Handle plan completion
 */
async function onPlanComplete() {
    requestAnalyzePlan(true);
}

/**
 * Scroll to step in Outputs tab
 */
function scrollToStepOutput(stepIndex) {
    // Switch to Outputs tab
    switchDetailTab('outputs');
    
    // Scroll to the step section
    setTimeout(() => {
        const stepEl = document.getElementById(`step-output-${stepIndex}`);
        if (stepEl) {
            stepEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 100);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => setupDetailPanelListeners());
} else {
    setupDetailPanelListeners();
}
