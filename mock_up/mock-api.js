// ============================================================
// Mock API Layer for Standalone Demo
// Overrides window.fetch BEFORE any other script loads.
// ============================================================

(function () {
    'use strict';

    const _realFetch = window.fetch.bind(window);

    // ── Inline language data (for file:// protocol support) ──

    const LANG_DATA = {
        en: {"_meta.name":"English","_meta.code":"en","error.analysis_failed":"Analysis generation failed","error.generic":"Error: {message}","error.no_result":"No result","error.request_failed":"Request failed","error.retry_failed":"Retry failed: {message}","error.code_gen_failed":"Code generation failed","status.analyzing":"Analyzing...","status.executing_plan":"Executing plan...","status.generating_code":"Generating code...","status.running":"Running...","status.completed":"Completed","status.error":"Error","status.regenerating":"LLM is regenerating...","status.executing":"{name} executing...","status.generating_answer":"Generating answer...","label.execution_plan":"Execution Plan","label.comprehensive":"Summary","label.comprehensive_code":"Summary Analysis Code","label.result":"Result","label.retry":"Retry","label.cancel":"Cancel","label.save":"Save","label.modified":"Modified","label.regenerate":"Regenerate","label.regenerate_analysis":"Regenerate Analysis","label.visualization":"{tool} Visualization","label.cot":"Show thinking process","label.entire_plan":"Entire Plan","label.cloning":"Cloning","label.transduction":"Transduction","label.analysis":"Analysis","label.preparation":"Preparation","label.execution":"Execution","label.week_timeline":"{weeks}-week Experiment Timeline","label.sgrna_distribution":"sgRNA Efficiency Score Distribution","label.settings_language":"Language","placeholder.step_question":"Ask a question about Step {num}...","placeholder.plan_question":"Ask a question about the entire Plan...","placeholder.edit_result":"Modify or supplement the result...","empty.plan_hint":"Send a message in Plan mode and\nthe analysis plan will appear here.","empty.graph_hint":"When a Plan is generated, the node graph\nwill appear here.","empty.graph_popout":"Graph is open in a separate window.","empty.code_hint":"Code will be auto-generated when a Step completes.","empty.outputs_hint":"Step results will appear here.","empty.step_no_result":"This Step's result is not yet available.","empty.no_code":"No code available.","tooltip.plan_ref":"Ask question referencing entire Plan","tooltip.retry":"Retry","tooltip.edit_result":"Edit result","tooltip.ask":"Ask question"},
        ko: {"_meta.name":"한국어","_meta.code":"ko","error.analysis_failed":"분석 생성 실패","error.generic":"오류: {message}","error.no_result":"결과 없음","error.request_failed":"요청 실패","error.retry_failed":"재시도 실패: {message}","error.code_gen_failed":"코드 생성 실패","status.analyzing":"분석 중...","status.executing_plan":"플랜 실행 중...","status.generating_code":"코드 생성 중...","status.running":"실행 중...","status.completed":"완료","status.error":"오류","status.regenerating":"LLM이 재생성 중...","status.executing":"{name} 실행 중...","status.generating_answer":"답변 생성 중...","label.execution_plan":"실행 계획","label.comprehensive":"종합","label.comprehensive_code":"종합 분석 코드","label.result":"결과","label.retry":"재시도","label.cancel":"취소","label.save":"저장","label.modified":"수정됨","label.regenerate":"재생성","label.regenerate_analysis":"분석 재생성","label.visualization":"{tool} 시각화","label.cot":"생각하는 과정 표시","label.entire_plan":"Plan 전체","label.cloning":"클로닝","label.transduction":"형질도입","label.analysis":"분석","label.preparation":"준비","label.execution":"실행","label.week_timeline":"{weeks}주 실험 타임라인","label.sgrna_distribution":"sgRNA 효율 점수 분포","label.settings_language":"언어","placeholder.step_question":"Step {num}에 대해 질문하세요...","placeholder.plan_question":"Plan 전체에 대해 질문하세요...","placeholder.edit_result":"결과를 수정하거나 보완하세요...","empty.plan_hint":"Plan 모드에서 메시지를 보내면\n분석 계획이 여기에 표시됩니다.","empty.graph_hint":"Plan이 생성되면 노드 그래프가\n여기에 표시됩니다.","empty.graph_popout":"Graph가 별도 창에서 열려 있습니다.","empty.code_hint":"Step 완료 시 코드가 자동 생성됩니다.","empty.outputs_hint":"Step 결과가 여기에 표시됩니다.","empty.step_no_result":"이 Step의 결과가 아직 없습니다.","empty.no_code":"코드가 없습니다.","tooltip.plan_ref":"Plan 전체 참고하여 질문","tooltip.retry":"재시도","tooltip.edit_result":"결과 수정","tooltip.ask":"질문하기"}
    };

    // ── Node manifest ──

    const NODE_MANIFEST = { files: [
        "nodes/math/math-helper.js",
        "nodes/analyze/analyze-node.js",
        "nodes/codegen/codegen-node.js",
        "nodes/composite/composite-node.js",
        "nodes/crispr/crispr-node.js",
        "nodes/data/data-node.js",
        "nodes/float/float-node.js",
        "nodes/image/image-node.js",
        "nodes/integer/integer-node.js",
        "nodes/math/add-node.js",
        "nodes/math/divide-node.js",
        "nodes/math/log-node.js",
        "nodes/math/multiply-node.js",
        "nodes/math/power-node.js",
        "nodes/math/sqrt-node.js",
        "nodes/math/subtract-node.js",
        "nodes/matrix3/matrix3-node.js",
        "nodes/matrix4/matrix4-node.js",
        "nodes/ncbi-gene/ncbi-gene-node.js",
        "nodes/observe/observe-node.js",
        "nodes/protocol/protocol-node.js",
        "nodes/pubmed/pubmed-node.js",
        "nodes/save/save-node.js",
        "nodes/step/step-node.js",
        "nodes/string/string-node.js",
        "nodes/table/table-node.js",
        "nodes/visualize/visualize-node.js"
    ]};

    // ── Mock plan data ──

    const PLAN_GOAL = 'T세포 고갈을 조절하는 유전자를 식별하기 위한 CRISPR 스크린 실험 계획';

    const PLAN_STEPS = [
        { id: 1, name: '관련 문헌 검색',        tool: 'pubmed_search',    description: 'T세포 고갈 관련 CRISPR 스크린 논문을 검색합니다.' },
        { id: 2, name: '후보 유전자 식별',      tool: 'ncbi_gene',        description: 'NCBI Gene 데이터베이스에서 후보 유전자 정보를 수집합니다.' },
        { id: 3, name: 'CRISPR 가이드 RNA 설계', tool: 'crispr_designer',  description: '후보 유전자에 대한 가이드 RNA를 설계합니다.' },
        { id: 4, name: '실험 프로토콜 작성',    tool: 'protocol_builder', description: 'CRISPR 스크린 실험 프로토콜을 작성합니다.' },
        { id: 5, name: '데이터 분석 코드 생성',  tool: 'code_gen',         description: '스크린 결과 분석을 위한 Python 코드를 생성합니다.' }
    ];

    const STEP_RESULTS = [
        {
            step: 1, tool: 'pubmed_search', success: true,
            thought: 'T세포 고갈 CRISPR 스크린 관련 최신 논문을 검색합니다.',
            action: 'PubMed API call: query="T cell exhaustion CRISPR screen", max_results=50',
            result: {
                title: 'Found 847 related papers',
                details: [
                    'Search query: T cell exhaustion CRISPR screen',
                    'Key papers: Chen et al. (2023) - Genome-wide CRISPR screen identifies PDCD1 regulators',
                    'Wei et al. (2024) - In vivo CRISPR screen of T cell exhaustion genes',
                    'Top candidate genes: TOX, PDCD1, LAG3, HAVCR2, TIGIT'
                ],
                query: 'T cell exhaustion CRISPR screen', max_results: 50,
                tokens: 2946, duration: '3.2s'
            }
        },
        {
            step: 2, tool: 'ncbi_gene', success: true,
            thought: '후보 유전자의 상세 정보를 NCBI Gene에서 조회합니다.',
            action: "NCBI Gene API call: genes=['TOX','PDCD1','LAG3','HAVCR2','TIGIT','CTLA4','CD28','EOMES']",
            result: {
                title: 'Retrieved info for 8 genes',
                details: [
                    'TOX: key exhaustion transcription factor, chr 8q12.1',
                    'PDCD1: PD-1 inhibitory receptor, chr 2q37.3',
                    'LAG3: inhibitory receptor, chr 12p13.32',
                    'HAVCR2: TIM-3 receptor, chr 5q33.3',
                    'TIGIT: co-inhibitory receptor, chr 3q13.31',
                    'CTLA4: co-inhibitory receptor, chr 2q33.2',
                    'CD28: co-stimulatory receptor, chr 2q33.2',
                    'EOMES: T-box transcription factor, chr 3p24.1'
                ],
                queried_genes: ['TOX','PDCD1','LAG3','HAVCR2','TIGIT','CTLA4','CD28','EOMES'],
                tokens: 2453, duration: '3.1s'
            }
        },
        {
            step: 3, tool: 'crispr_designer', success: true,
            thought: '8개 후보 유전자에 대한 sgRNA를 설계합니다.',
            action: 'CRISPR guide RNA design for 8 target genes',
            result: {
                title: 'Designed 24 guide RNAs for 8 genes',
                details: [
                    'TOX: 3 sgRNAs designed (efficiency: 0.85, 0.82, 0.79)',
                    'PDCD1: 3 sgRNAs designed (efficiency: 0.91, 0.88, 0.84)',
                    'LAG3: 3 sgRNAs designed (efficiency: 0.87, 0.83, 0.80)',
                    'Total: 24 guides, avg efficiency: 0.84'
                ],
                target_genes: ['TOX','PDCD1','LAG3','HAVCR2','TIGIT','CTLA4','CD28','EOMES'],
                has_graph: false, duration: '4.5s'
            }
        },
        {
            step: 4, tool: 'protocol_builder', success: true,
            thought: 'CRISPR 스크린 실험의 전체 프로토콜을 작성합니다.',
            action: 'Protocol generation for CRISPR screen experiment',
            result: {
                title: 'CRISPR Screen Protocol Generated',
                details: [
                    'Phase 1: sgRNA library cloning (Week 1-2)',
                    'Phase 2: Lentiviral production & T cell transduction (Week 3-4)',
                    'Phase 3: T cell exhaustion induction with chronic stimulation (Week 5-8)',
                    'Phase 4: FACS sorting of exhausted vs functional T cells (Week 9)',
                    'Phase 5: NGS library preparation and sequencing (Week 10-11)',
                    'Phase 6: Data analysis and hit identification (Week 12)'
                ],
                experiment_type: 'CRISPR Screen', duration_weeks: 12, duration: '2.8s'
            }
        },
        {
            step: 5, tool: 'code_gen', success: true,
            thought: '스크린 결과 분석을 위한 Python 코드를 생성합니다.',
            action: 'Python code generation for CRISPR screen analysis',
            result: {
                language: 'python',
                code: [
                    'import pandas as pd',
                    'import numpy as np',
                    'from scipy import stats',
                    '',
                    'def analyze_crispr_screen(counts_file):',
                    '    """Analyze CRISPR screen results."""',
                    '    counts = pd.read_csv(counts_file)',
                    '    total_reads = counts.sum(axis=0)',
                    '    normalized = counts.div(total_reads) * 1e6  # CPM',
                    '',
                    '    exhausted = [c for c in counts.columns if "exhausted" in c]',
                    '    functional = [c for c in counts.columns if "functional" in c]',
                    '',
                    '    lfc = np.log2(normalized[exhausted].mean(axis=1) + 1) \\',
                    '        - np.log2(normalized[functional].mean(axis=1) + 1)',
                    '',
                    '    pvalues = []',
                    '    for idx in counts.index:',
                    '        _, p = stats.ttest_ind(',
                    '            normalized.loc[idx, exhausted],',
                    '            normalized.loc[idx, functional])',
                    '        pvalues.append(p)',
                    '',
                    '    results = pd.DataFrame({',
                    "        'sgRNA': counts.index,",
                    "        'log2FC': lfc.values,",
                    "        'pvalue': pvalues",
                    '    })',
                    "    results['padj'] = stats.false_discovery_control(results['pvalue'])",
                    "    results = results.sort_values('padj')",
                    '',
                    '    print(f"Total sgRNAs analyzed: {len(results)}")',
                    '    print(f"Significant hits (padj < 0.05): {(results[\'padj\'] < 0.05).sum()}")',
                    '    print(results.head(10).to_string())',
                    '    return results',
                    '',
                    '# Demo',
                    'np.random.seed(42)',
                    'mock = pd.DataFrame({',
                    "    'exhausted_rep1': np.random.negative_binomial(5, 0.01, 24),",
                    "    'exhausted_rep2': np.random.negative_binomial(5, 0.01, 24),",
                    "    'functional_rep1': np.random.negative_binomial(10, 0.01, 24),",
                    "    'functional_rep2': np.random.negative_binomial(10, 0.01, 24),",
                    '})',
                    "mock.to_csv('mock_counts.csv', index=False)",
                    "results = analyze_crispr_screen('mock_counts.csv')",
                    'print("\\nAnalysis complete.")'
                ].join('\n'),
                task: 'Generate CRISPR screen analysis code',
                execution: {
                    success: true,
                    stdout: 'Total sgRNAs analyzed: 24\nSignificant hits (padj < 0.05): 6\n   sgRNA    log2FC    pvalue      padj\n0      0 -1.234567  0.003421  0.013684\n1      1  0.876543  0.012345  0.024690\n2      2 -0.654321  0.023456  0.031275\n3      3  1.345678  0.034567  0.034567\n4      4 -0.987654  0.043210  0.034568\n5      5  0.543210  0.001234  0.009872\n\nAnalysis complete.',
                    stderr: '', returncode: 0
                },
                fix_attempts: 0, duration: '5.2s'
            }
        }
    ];

    const PLAN_ANALYSIS = [
        '## 연구 목표\n',
        'T세포 고갈을 조절하는 유전자를 식별하기 위한 CRISPR 스크린 실험 계획\n\n',
        '## 실험 설계 요약\n\n',
        '본 계획은 5단계로 구성됩니다:\n\n',
        '1. **문헌 검색**: PubMed를 통한 847편의 관련 논문 분석\n',
        '2. **후보 유전자 선정**: TOX, PDCD1, LAG3 등 8개 핵심 유전자 확인\n',
        '3. **sgRNA 설계**: 24개의 가이드 RNA (평균 효율 0.84)\n',
        '4. **실험 프로토콜**: 12주 계획 (라이브러리 구축 → 형질전환 → 고갈 유도 → FACS → NGS → 분석)\n',
        '5. **분석 코드**: Python 기반 통계 분석 파이프라인\n\n',
        '## 핵심 발견\n\n',
        '- **주요 후보 유전자**: TOX (고갈 핵심 전사인자), PDCD1 (PD-1 억제 수용체), LAG3, HAVCR2\n',
        '- **통계적 유의성**: 24개 sgRNA 중 6개에서 유의미한 enrichment 확인 (padj < 0.05)\n\n',
        '## 수학적 모델\n\n',
        'Log2 fold change 계산:\n',
        '$$\\text{LFC} = \\log_2\\left(\\frac{\\text{CPM}_{\\text{exhausted}} + 1}{\\text{CPM}_{\\text{functional}} + 1}\\right)$$\n\n',
        '## 참고 문헌\n\n',
        '- Chen et al. (2023) - Genome-wide CRISPR screen\n',
        '- Wei et al. (2024) - In vivo CRISPR screen'
    ].join('');

    const AGENT_RESPONSE = [
        '## T세포 고갈 (T Cell Exhaustion) 메커니즘\n\n',
        'T세포 고갈은 만성 감염이나 종양 환경에서 T세포가 점진적으로 기능을 상실하는 과정입니다.\n\n',
        '### 주요 특징\n\n',
        '1. **억제 수용체 발현 증가**\n',
        '   - PD-1 (PDCD1)\n',
        '   - LAG-3\n',
        '   - TIM-3 (HAVCR2)\n',
        '   - TIGIT\n\n',
        '2. **사이토카인 분비 감소**\n',
        '   - IFN-γ ↓\n',
        '   - TNF-α ↓\n',
        '   - IL-2 ↓\n\n',
        '3. **전사 프로그램 변화**\n',
        '   - TOX 상향 조절\n',
        '   - T-bet 하향 조절\n',
        '   - Eomes 상향 조절\n\n',
        '### 분자 메커니즘\n\n',
        '고갈된 T세포에서의 에너지 대사 변화:\n\n',
        '$$E_{\\text{glycolysis}} = \\sum_{i=1}^{n} k_i \\cdot [S_i] \\cdot \\frac{V_{\\max}}{K_m + [S_i]}$$\n\n',
        '### 코드 예시\n\n',
        '```python\n',
        'import scanpy as sc\n\n',
        '# Load single-cell data\n',
        "adata = sc.read_h5ad('t_cells.h5ad')\n\n",
        '# Identify exhaustion markers\n',
        "markers = ['PDCD1', 'LAG3', 'HAVCR2', 'TOX']\n",
        "sc.pl.dotplot(adata, markers, groupby='cluster')\n",
        '```\n\n',
        '### 치료적 접근\n\n',
        '| 전략 | 표적 | 단계 |\n',
        '|------|------|------|\n',
        '| Anti-PD-1 | PDCD1 | FDA 승인 |\n',
        '| Anti-LAG-3 | LAG3 | 임상 3상 |\n',
        '| TOX 억제 | TOX | 전임상 |\n\n',
        '> **핵심 포인트**: T세포 고갈은 가역적 과정이며, 면역 체크포인트 차단을 통해 부분적으로 회복될 수 있습니다.'
    ].join('');

    // ── In-memory state ──

    const now = new Date().toISOString();

    const planCompletePayload = {
        goal: PLAN_GOAL,
        steps: PLAN_STEPS.map(s => ({ id: s.id, name: s.name, description: s.description, status: 'pending' })),
        results: STEP_RESULTS
    };

    let conversations = {};

    let nextId = 100;
    let systemPrompt = 'You are a helpful biomedical research assistant.';
    let settings = { temperature: 0.7, max_length: 16384, top_k: 50, max_images: 5, max_context: 32768 };

    // ── Helpers ──

    function jsonResp(data, status) {
        return new Response(JSON.stringify(data), {
            status: status || 200,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    function convList() {
        return Object.values(conversations)
            .sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || ''))
            .map(c => ({ id: c.id, title: c.title, updated_at: c.updated_at, created_at: c.created_at }));
    }

    function sseEvent(obj) {
        return 'data: ' + JSON.stringify(obj) + '\n\n';
    }

    function makeSSEStream(events, signal) {
        const encoder = new TextEncoder();
        let cancelled = false;
        if (signal) signal.addEventListener('abort', () => { cancelled = true; });

        return new ReadableStream({
            async start(controller) {
                for (let i = 0; i < events.length; i++) {
                    if (cancelled) {
                        try { controller.close(); } catch (_) {}
                        return;
                    }
                    const ev = events[i];
                    const delay = ev._delay || 30;
                    await new Promise(r => setTimeout(r, delay));
                    if (cancelled) { try { controller.close(); } catch (_) {} return; }
                    const { _delay, ...data } = ev;
                    controller.enqueue(encoder.encode(sseEvent(data)));
                }
                try { controller.close(); } catch (_) {}
            }
        });
    }

    // ── SSE event builders ──

    function buildPlanSSE() {
        const events = [];
        const introTokens = '계획을 수립하겠습니다.\n\n';
        for (const ch of introTokens) {
            events.push({ token: ch, _delay: 15 });
        }

        events.push({
            tool_call: {
                name: 'create_plan',
                arguments: { goal: PLAN_GOAL, steps: PLAN_STEPS.map(s => ({ name: s.name, tool: s.tool, description: s.description })) },
                status: 'running'
            },
            _delay: 100
        });

        events.push({
            tool_result: {
                success: true, tool: 'create_plan',
                result: {
                    goal: PLAN_GOAL,
                    total_steps: PLAN_STEPS.length,
                    steps: PLAN_STEPS.map(s => ({ id: s.id, name: s.name, description: s.description, status: 'pending' })),
                    current_step: 0
                }
            },
            _delay: 200
        });

        for (const r of STEP_RESULTS) {
            events.push({ step_start: { step: r.step }, _delay: 400 });
            events.push({ tool_result: { ...r }, _delay: 800 });
        }

        events.push({
            done: true,
            plan_complete: planCompletePayload,
            _delay: 300
        });

        return events;
    }

    function buildAgentSSE() {
        const events = [];
        const words = AGENT_RESPONSE.split(/(?<=\s)/);
        for (const w of words) {
            events.push({ token: w, _delay: 20 });
        }
        events.push({ done: true, _delay: 50 });
        return events;
    }

    function buildQuestionSSE() {
        const answer = '해당 단계의 결과를 분석해 보겠습니다.\n\n이 단계에서는 목표한 작업이 성공적으로 수행되었으며, 결과 데이터의 품질도 양호합니다. 추가적인 분석이 필요하시면 말씀해 주세요.';
        const events = [];
        const words = answer.split(/(?<=\s)/);
        for (const w of words) {
            events.push({ token: w, _delay: 20 });
        }
        events.push({ done: true, _delay: 50 });
        return events;
    }

    // ── Route matching ──

    function extractPath(url) {
        if (typeof url !== 'string') url = url.toString();
        try { return new URL(url).pathname; } catch (_) {}
        return url.split('?')[0];
    }

    function matchConvId(path) {
        const m = path.match(/\/api\/conversation\/([^/]+)/);
        return m ? m[1] : null;
    }

    async function parseBody(options) {
        if (!options || !options.body) return {};
        try {
            if (typeof options.body === 'string') return JSON.parse(options.body);
            if (options.body instanceof FormData) return { _formData: options.body };
            if (options.body instanceof ReadableStream) {
                const reader = options.body.getReader();
                let result = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    result += new TextDecoder().decode(value);
                }
                return JSON.parse(result);
            }
            return JSON.parse(options.body);
        } catch (_) { return {}; }
    }

    // ── Main fetch override ──

    window.fetch = async function (url, options) {
        const path = extractPath(url);
        const method = ((options && options.method) || 'GET').toUpperCase();

        // Language files: serve from inline data, fallback to real fetch
        const langMatch = path.match(/language\/(\w+)\.json$/);
        if (langMatch) {
            const code = langMatch[1];
            if (LANG_DATA[code]) return jsonResp(LANG_DATA[code]);
            try { return await _realFetch(url, options); } catch (_) { return jsonResp({}); }
        }

        // ── API routes ──

        // Node manifest
        if (path === '/api/node-manifest') return jsonResp(NODE_MANIFEST);

        // Model info
        if (path === '/api/model') return jsonResp({ model: 'Mock-LLM-7B (Demo)' });

        // Conversations list
        if (path === '/api/conversations') return jsonResp(convList());

        // Settings
        if (path === '/api/settings') {
            if (method === 'POST') { Object.assign(settings, await parseBody(options)); return jsonResp({ ok: true }); }
            return jsonResp(settings);
        }

        // System prompt
        if (path === '/api/system_prompt') {
            if (method === 'POST') { const b = await parseBody(options); systemPrompt = b.prompt || systemPrompt; return jsonResp({ ok: true }); }
            return jsonResp({ prompt: systemPrompt });
        }

        // New conversation
        if (path === '/api/new' && method === 'POST') {
            const id = 'mock-' + (nextId++);
            conversations[id] = {
                id, title: 'New Conversation',
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                messages: []
            };
            return jsonResp({ id, title: 'New Conversation' });
        }

        // Stop generation
        if (path === '/api/stop') return jsonResp({ ok: true });

        // Chat (SSE)
        if (path === '/api/chat' && method === 'POST') {
            const body = await parseBody(options);
            const convId = body.conversation_id;
            const mode = body.mode || 'agent';
            const userMsg = body.message || '';

            if (convId && conversations[convId]) {
                conversations[convId].messages.push({ role: 'user', content: userMsg });
                conversations[convId].updated_at = new Date().toISOString();
                if (conversations[convId].title === 'New Conversation' && userMsg.length > 0) {
                    conversations[convId].title = userMsg.substring(0, 40) + (userMsg.length > 40 ? '...' : '');
                }
            }

            const events = mode === 'plan' ? buildPlanSSE() : buildAgentSSE();
            const stream = makeSSEStream(events, options && options.signal);

            // Append assistant message after stream
            if (convId && conversations[convId]) {
                const assistantContent = mode === 'plan'
                    ? '[PLAN_COMPLETE]' + JSON.stringify(planCompletePayload)
                    : AGENT_RESPONSE;
                setTimeout(() => {
                    conversations[convId].messages.push({ role: 'assistant', content: assistantContent });
                }, events.length * 50 + 500);
            }

            return new Response(stream, {
                status: 200,
                headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' }
            });
        }

        // Conversation CRUD
        const convId = matchConvId(path);
        if (convId) {
            if (method === 'DELETE') {
                delete conversations[convId];
                return jsonResp({ ok: true });
            }
            if (path.endsWith('/rename') && method === 'POST') {
                const b = await parseBody(options);
                if (conversations[convId]) conversations[convId].title = b.title || conversations[convId].title;
                return jsonResp({ ok: true });
            }
            if (path.endsWith('/clear') && method === 'POST') {
                if (conversations[convId]) conversations[convId].messages = [];
                return jsonResp({ ok: true });
            }
            if (path.endsWith('/truncate') && method === 'POST') {
                if (conversations[convId]) {
                    const b = await parseBody(options);
                    const idx = b.index || conversations[convId].messages.length - 2;
                    conversations[convId].messages = conversations[convId].messages.slice(0, idx);
                }
                return jsonResp({ ok: true });
            }
            if (method === 'GET' && conversations[convId]) {
                return jsonResp(conversations[convId]);
            }
            return jsonResp({ error: 'Conversation not found' }, 404);
        }

        // Tool call
        if (path === '/api/tool_call' && method === 'POST') {
            const b = await parseBody(options);
            if (b.tool === 'analyze_plan') {
                return jsonResp({
                    success: true, tool: 'analyze_plan',
                    result: { analysis: PLAN_ANALYSIS, goal: PLAN_GOAL, total_steps: PLAN_STEPS.length, current_step: 0 }
                });
            }
            if (b.tool === 'code_gen') {
                return jsonResp(STEP_RESULTS[4]);
            }
            return jsonResp({ success: true, tool: b.tool, result: { title: 'Mock result', details: ['Demo result for ' + b.tool] } });
        }

        // Execute code
        if (path === '/api/execute_code' && method === 'POST') {
            return jsonResp({
                success: true,
                stdout: 'Total sgRNAs analyzed: 24\nSignificant hits (padj < 0.05): 6\n\nAnalysis complete.',
                stderr: '', returncode: 0, figures: [], tables: []
            });
        }

        // Replan
        if (path === '/api/replan' && method === 'POST') {
            return jsonResp({ ok: true });
        }

        // Update plan analysis
        if (path === '/api/update_plan_analysis' && method === 'POST') {
            return jsonResp({ ok: true });
        }

        // Step question (SSE)
        if (path === '/step_question' && method === 'POST') {
            const body = await parseBody(options);
            const convId2 = body.conversation_id;
            if (convId2 && conversations[convId2]) {
                conversations[convId2].messages.push({ role: 'user', content: body.question || '' });
                setTimeout(() => {
                    conversations[convId2].messages.push({ role: 'assistant', content: '해당 단계의 결과를 분석해 보겠습니다.\n\n이 단계에서는 목표한 작업이 성공적으로 수행되었으며, 결과 데이터의 품질도 양호합니다. 추가적인 분석이 필요하시면 말씀해 주세요.' });
                }, 2000);
            }
            const stream = makeSSEStream(buildQuestionSSE(), options && options.signal);
            return new Response(stream, {
                status: 200,
                headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' }
            });
        }

        // Retry step (SSE)
        if (path === '/retry_step' && method === 'POST') {
            const stream = makeSSEStream(buildQuestionSSE(), options && options.signal);
            return new Response(stream, {
                status: 200,
                headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' }
            });
        }

        // Upload
        if (path === '/api/data/upload' && method === 'POST') {
            const fd = options && options.body;
            const files = [];
            if (fd instanceof FormData) {
                for (const [key, file] of fd.entries()) {
                    if (file instanceof File) {
                        files.push({ name: file.name, path: '/uploads/' + file.name, type: file.type, url: URL.createObjectURL(file) });
                    }
                }
            }
            return jsonResp({ success: true, files });
        }

        // Outputs
        if (path.startsWith('/api/outputs/')) {
            return jsonResp({ files: [] });
        }

        // Catch-all: pass through to real fetch (for scripts, CSS, etc.)
        return _realFetch(url, options);
    };

    console.log('%c[Mock API] Demo mode active', 'color: #4CAF50; font-weight: bold;');
})();
