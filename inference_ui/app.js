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
let editingIndex = -1;  // -1 = 수정 모드 아님
let pendingFiles = [];  // 첨부 파일 배열 { type: 'image'|'audio', name, data }
let scrollLockUntil = 0;  // 스크롤 잠금 해제 시간 (timestamp)

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

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    // Apply saved theme
    setTheme(getTheme());
    
    // Apply saved background image, blur, and opacity
    applyBgImage();
    applyBgBlur();
    applyBgOpacity();
    
    // Load sidebar state from localStorage (transition 없이 초기 적용)
    const savedSidebarState = localStorage.getItem('sidebarCollapsed');
    if (savedSidebarState === 'true') {
        sidebarCollapsed = true;
        sidebar.style.transition = 'none';
        mainContent.style.transition = 'none';
        mainContent.style.marginLeft = '50px';
        sidebar.classList.add('collapsed');
        
        // 다음 프레임에서 transition 속성 완전 제거
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                sidebar.style.removeProperty('transition');
                mainContent.style.removeProperty('transition');
            });
        });
    }
    
    // Setup event listeners
    setupEventListeners();
    
    // Load conversations list
    await loadConversations();
    
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
    
    // Enter to send
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !isStreaming) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // File upload
    setupFileUpload();
    
    // 휠 이벤트 - 스크롤 잠금/해제 (사용자의 의도적인 스크롤만)
    messagesContainer.addEventListener('wheel', (e) => {
        if (e.deltaY < 0) {
            // 위로 스크롤 → 무한 잠금 (사용자가 하단으로 스크롤할 때까지)
            scrollLockUntil = Infinity;
        } else if (e.deltaY > 0) {
            // 아래로 스크롤 → 하단 근처면 잠금 해제 (자동 스크롤 재활성화)
            const threshold = 150;  // 하단에서 150px 이내
            const distanceFromBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight;
            if (distanceFromBottom < threshold) {
                scrollLockUntil = 0;
            }
        }
    });
    
    // 스크롤 이벤트 - 버튼 표시/숨김만 (잠금 해제는 wheel에서만)
    messagesContainer.addEventListener('scroll', () => {
        const threshold = 200;
        const distanceFromBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight;
        
        if (distanceFromBottom > threshold) {
            scrollToBottomBtn.classList.add('visible');
        } else {
            scrollToBottomBtn.classList.remove('visible');
            // 잠금 해제는 wheel 이벤트에서만 처리 (프로그래매틱 스크롤과 구분)
        }
    });
    
    // 하단 이동 버튼 클릭
    scrollToBottomBtn.addEventListener('click', () => {
        scrollLockUntil = 0;
        scrollToBottom(true);
    });
    
    // Input area hover 시 버튼 선명하게
    const inputArea = document.getElementById('dropZone');
    inputArea.addEventListener('mouseenter', () => {
        scrollToBottomBtn.classList.add('input-hover');
    });
    inputArea.addEventListener('mouseleave', () => {
        scrollToBottomBtn.classList.remove('input-hover');
    });
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

function getMaxImages() {
    return parseInt(localStorage.getItem('maxImages') || '5');
}

function handleFiles(files) {
    const maxImages = getMaxImages();
    
    for (const file of files) {
        // Check image limit before adding
        if (file.type.startsWith('image/')) {
            const currentImageCount = pendingFiles.filter(f => f.type === 'image').length;
            if (currentImageCount >= maxImages) {
                alert(`Maximum ${maxImages} images allowed. You can change this in Settings.`);
                return;
            }
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            if (file.type.startsWith('image/')) {
                // Double-check limit in async callback
                const currentImageCount = pendingFiles.filter(f => f.type === 'image').length;
                if (currentImageCount < maxImages) {
                    pendingFiles.push({ type: 'image', name: file.name, data: e.target.result });
                    renderFilePreviews();
                }
            } else if (file.type.startsWith('audio/')) {
                pendingFiles.push({ type: 'audio', name: file.name, data: e.target.result });
                renderFilePreviews();
            }
        };
        reader.readAsDataURL(file);
    }
}

function renderFilePreviews() {
    const container = document.getElementById('filePreviewContainer');
    if (pendingFiles.length === 0) {
        container.innerHTML = '';  // :empty CSS로 자동 숨김
        return;
    }
    
    // 파일 미리보기 렌더링 (:empty가 아니므로 자동 표시)
    container.innerHTML = pendingFiles.map((f, i) => `
        <div class="file-preview ${f.type}">
            ${f.type === 'image' 
                ? `<img src="${f.data}" alt="${escapeHtml(f.name)}">` 
                : `<span class="file-icon">🎵</span><span class="file-name">${escapeHtml(f.name)}</span>`
            }
            <button class="file-remove" onclick="removeFile(${i})" title="Remove">×</button>
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
    
    // 메인 영역 margin-left 설정
    const sidebarWidth = sidebarCollapsed ? 50 : 280;
    mainContent.style.marginLeft = sidebarWidth + 'px';
    
    // 클래스 토글 (transform은 CSS에서 처리)
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
    // 다른 채팅으로 이동 시 생성 중이면 중단 (현재까지 내용은 서버에서 자동 저장됨)
    if (isStreaming && currentConversationId !== id) {
        stopGeneration();
        // 잠시 대기하여 서버가 partial response를 저장할 시간 확보
        await new Promise(r => setTimeout(r, 300));
    }
    
    // 채팅 로드 시 자동 스크롤 활성화 (기본 동작)
    scrollLockUntil = 0;
    
    try {
        const response = await fetch(`/api/conversation/${id}`);
        if (!response.ok) throw new Error('Failed to load conversation');
        
        const conversation = await response.json();
        currentConversationId = id;
        
        renderConversationsList();
        renderMessages(conversation.messages || []);
    } catch (error) {
        console.error('Error loading conversation:', error);
    }
}

async function createNewChat() {
    // 생성 중이면 중단 (현재까지 내용은 서버에서 자동 저장됨)
    if (isStreaming) {
        stopGeneration();
        await new Promise(r => setTimeout(r, 300));
    }
    
    // 새 채팅 시 자동 스크롤 활성화
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
    
    if (!isUser && message.content) {
        // Parse all special tokens from content
        const parsed = parseSpecialTokens(message.content);
        
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
        
        contentHTML = filesHTML + `<div class="answer-content">${renderMarkdown(textContent)}</div>`;
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

function parseSpecialTokens(content) {
    let result = {
        think: null,
        tools: null,
        toolCalls: null,
        toolResults: null,
        toolContent: null,
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
    
    // [TOOL_CALLS] (단독 토큰, 뒤에 내용이 따라옴)
    const toolCallsMatch = result.answer.match(/\[TOOL_CALLS\]([\s\S]*?)(?=\[(?!\/)|$)/);
    if (toolCallsMatch) {
        result.toolCalls = toolCallsMatch[1].trim();
        result.answer = result.answer.replace(/\[TOOL_CALLS\][\s\S]*?(?=\[(?!\/)|$)/, '');
    }
    
    // [TOOL_CONTENT] (단독 토큰)
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
    
    // [ARGS]...(JSON until next token or end)
    const argsMatch = result.answer.match(/\[ARGS\]([\s\S]*?)(?=\[(?!\/)|$)/);
    if (argsMatch) {
        result.args = argsMatch[1].trim();
        result.answer = result.answer.replace(/\[ARGS\][\s\S]*?(?=\[(?!\/)|$)/, '');
    }
    
    // [CALL_ID]...(ID until next token or whitespace)
    const callIdMatch = result.answer.match(/\[CALL_ID\](\S+)/);
    if (callIdMatch) {
        result.callId = callIdMatch[1].trim();
        result.answer = result.answer.replace(/\[CALL_ID\]\S+/, '');
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
                <span class="cot-label">생각하는 과정 표시</span>
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

function createToolCallsHTML(content) {
    return `
        <div class="special-token-container tool-calls-container">
            <div class="special-badge tool-calls-badge">
                <span class="special-icon">⚡</span>
                <span class="special-label">Tool Call</span>
            </div>
            <div class="special-content"><pre>${escapeHtml(content)}</pre></div>
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
    
    // 메시지 전송 시 자동 스크롤 활성화
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
        const fileLabels = files.map(f => f.type === 'image' ? `[Image: ${f.name}]` : `[Audio: ${f.name}]`).join(' ');
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
    
    try {
        // Build request body with files
        const requestBody = {
            conversation_id: currentConversationId,
            message: content
        };
        if (files.length > 0) {
            requestBody.files = files;
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
        let fullContent = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.token) {
                            fullContent += data.token;
                            updateAssistantMessage(contentDiv, fullContent);
                        }
                        
                        if (data.done) {
                            // Final update with all special tokens
                            const parsed = parseSpecialTokens(fullContent);
                            let finalHTML = '';
                            
                            if (parsed.think) {
                                finalHTML += createCoTHTML(parsed.think);
                            }
                            if (parsed.tools) {
                                finalHTML += createToolsHTML(parsed.tools);
                            }
                            if (parsed.toolCalls) {
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
                            renderMath();
                            
                            // Cancel stream and exit loop
                            await reader.cancel();
                            scrollToBottom();
                            await loadConversation(currentConversationId);
                            await loadConversations();  // Refresh sidebar to update title
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
        
    } catch (error) {
        if (error.name === 'AbortError') {
            // User stopped generation - keep partial response
            console.log('Generation stopped by user');
            // Wait for server to save partial response before reloading
            await new Promise(r => setTimeout(r, 500));
            // Reload to show Edit/Regenerate buttons and update sidebar title
            await loadConversation(currentConversationId);
            await loadConversations();
        } else {
            console.error('Error sending message:', error);
            contentDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        }
    } finally {
        isStreaming = false;
        currentAbortController = null;
        setStreamingUI(false);
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
    
    let html = '';
    
    // Show thinking indicator during streaming
    if (parsed.think) {
        const preview = parsed.think.substring(0, 80).replace(/\n/g, ' ');
        html += `<div class="cot-container">
            <button class="cot-toggle" onclick="toggleSpecialToken(this)">
                <span class="cot-icon">✦</span>
                <span class="cot-label" style="color: var(--text-cot);">생각하는 과정 표시</span>
                <span class="cot-arrow">▼</span>
            </button>
        </div>`;
    }
    
    // Show tool calls indicator during streaming
    if (parsed.toolCalls) {
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
    
    // Render main answer with placeholders
    let answerHTML = renderMarkdown(parsed.answer);
    answerHTML = answerHTML.replace(/\{\{IMG_PLACEHOLDER\}\}/g, createImgPlaceholder());
    answerHTML = answerHTML.replace(/\{\{AUDIO_PLACEHOLDER\}\}/g, createAudioPlaceholder());
    
    html += `<div class="answer-content">${answerHTML}</div>`;
    html += '<span class="streaming-indicator"><span class="streaming-dot"></span><span class="streaming-dot"></span><span class="streaming-dot"></span></span>';
    
    contentDiv.innerHTML = html;
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
    // 시간 기반 잠금: 잠금 시간이 지났거나 force면 스크롤
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
    
    // 메시지 요소 찾기
    const messageEl = document.querySelector(`.message[data-index="${index}"]`);
    if (!messageEl) return;
    
    // 이미 편집 중이면 무시
    if (messageEl.classList.contains('editing')) return;
    
    // 편집 모드 진입
    editingIndex = index;
    messageEl.classList.add('editing');
    
    const contentDiv = messageEl.querySelector('.answer-content');
    
    // 현재 너비 저장 및 min-width로 설정하여 너비 유지
    const currentWidth = contentDiv.offsetWidth;
    contentDiv.style.minWidth = currentWidth + 'px';
    
    const originalContent = message.content;
    
    // textarea로 교체
    contentDiv.innerHTML = `
        <textarea class="edit-textarea">${escapeHtml(originalContent)}</textarea>
        <div class="edit-actions">
            <button class="edit-btn save" onclick="saveEdit(${index})">Save & Send</button>
            <button class="edit-btn cancel" onclick="cancelEdit(${index})">Cancel</button>
        </div>
    `;
    
    const textarea = contentDiv.querySelector('.edit-textarea');
    
    // 내용에 맞게 높이 자동 조절
    function autoResize() {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }
    autoResize();
    textarea.addEventListener('input', autoResize);
    
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    
    // Enter로 저장, ESC로 취소
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
            // UI 먼저 업데이트 (이전 메시지들 삭제 반영)
            await loadConversation(currentConversationId);
        }
        
        // 그 다음 새 메시지 전송
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
        // 간단한 피드백 (버튼 아이콘 변경)
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
    
    // 생성 중이면 먼저 중지
    if (isStreaming) {
        stopGeneration();
        await new Promise(r => setTimeout(r, 500));  // 잠시 대기
    }
    
    // 확인 다이얼로그
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
        // 현재 테마 버튼 강조
        const currentTheme = getTheme();
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === currentTheme);
        });
        // 배경 이미지 미리보기 업데이트
        updateBgPreview();
        // 블러 슬라이더 값 업데이트
        const blur = getBgBlur();
        document.getElementById('settingBlur').value = blur;
        document.getElementById('blurValue').textContent = blur;
        // 투명도 슬라이더 값 업데이트
        const opacity = getBgOpacity();
        document.getElementById('settingOpacity').value = opacity;
        document.getElementById('opacityValue').textContent = opacity;
        
        document.getElementById('settingUserName').value = getUserDisplayName();
        document.getElementById('settingTemperature').value = data.temperature || 1.0;
        document.getElementById('settingMaxLength').value = data.max_length || 32768;
        document.getElementById('settingTopK').value = data.top_k || 50;
        
        // Max Images 슬라이더 값 업데이트
        const maxImages = getMaxImages();
        document.getElementById('settingMaxImages').value = maxImages;
        document.getElementById('maxImagesValue').textContent = maxImages;
        
        // Max Context 값 업데이트
        const maxContext = data.max_context || 32768;
        document.getElementById('settingMaxContext').value = maxContext;
        document.getElementById('maxContextWarning').style.display = 'none';
        
        openModal('settingsModal');
    } catch (error) {
        console.error('Error getting settings:', error);
    }
});

// 테마 버튼 클릭 이벤트 (즉시 적용 및 저장)
document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // 모든 버튼에서 active 제거
        document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
        // 클릭된 버튼에 active 추가
        btn.classList.add('active');
        // 테마 적용
        setTheme(btn.dataset.theme);
    });
});

// 배경 이미지 파일 선택 이벤트
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

// 블러 슬라이더 이벤트 (실시간 적용)
document.getElementById('settingBlur').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('blurValue').textContent = value;
    setBgBlur(value);
});

// 투명도 슬라이더 이벤트 (실시간 적용)
document.getElementById('settingOpacity').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('opacityValue').textContent = value;
    setBgOpacity(value);
});

// Max Images 슬라이더 이벤트
document.getElementById('settingMaxImages').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('maxImagesValue').textContent = value;
});

// Max Context 입력 이벤트 (256k 초과 경고)
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
    const maxImages = parseInt(document.getElementById('settingMaxImages').value);
    let maxContext = parseInt(document.getElementById('settingMaxContext').value);
    
    // Cap max_context at 256k
    if (maxContext > 262144) {
        maxContext = 262144;
    }
    if (maxContext < 1024) {
        maxContext = 1024;
    }
    
    // Save user name to localStorage
    setUserDisplayName(userName);
    
    // Save maxImages to localStorage
    localStorage.setItem('maxImages', maxImages.toString());
    
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
