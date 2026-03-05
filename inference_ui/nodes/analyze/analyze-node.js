// ============================================
// Analyze Plan Tool Node
// ============================================

NodeRegistry.register('analyze', {
    label: 'Analyze Plan',
    category: 'Tool',
    allowRef: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' },
        { name: 'out', dir: 'out', type: 'any' }
    ],

    defaultConfig: {
        title: 'Analyze Plan',
        tool: 'analyze_plan',
        menuTag: {
            en: 'Analysis', ko: '분석', ja: '分析', zh: '分析',
            fr: 'Analyse', de: 'Analyse', es: 'Análisis', it: 'Analisi',
            pt: 'Análise', nl: 'Analyse', ru: 'Анализ', ar: 'تحليل',
            hi: 'विश्लेषण', tr: 'Analiz', pl: 'Analiza', cs: 'Analýza',
            sv: 'Analys', da: 'Analyse', no: 'Analyse', fi: 'Analyysi',
            el: 'Ανάλυση', hu: 'Elemzés', ro: 'Analiză', uk: 'Аналіз',
            vi: 'Phân tích', th: 'การวิเคราะห์', id: 'Analisis'
        },
        description: {
            en: 'Analyze execution plan results and generate insights',
            ko: '실행 계획 결과를 분석하고 인사이트 생성',
            ja: '実行計画の結果を分析し、インサイトを生成',
            zh: '分析执行计划结果并生成洞察',
            fr: 'Analyser les résultats du plan d\'exécution et générer des insights',
            de: 'Ausführungsplanergebnisse analysieren und Erkenntnisse generieren',
            es: 'Analizar resultados del plan de ejecución y generar insights',
            it: 'Analizzare i risultati del piano di esecuzione e generare approfondimenti',
            pt: 'Analisar resultados do plano de execução e gerar insights',
            nl: 'Resultaten van het uitvoeringsplan analyseren en inzichten genereren',
            ru: 'Анализ результатов плана выполнения и генерация выводов',
            ar: 'تحليل نتائج خطة التنفيذ وتوليد الرؤى',
            hi: 'निष्पादन योजना के परिणामों का विश्लेषण करें और अंतर्दृष्टि उत्पन्न करें',
            tr: 'Yürütme planı sonuçlarını analiz edin ve içgörüler oluşturun',
            pl: 'Analizuj wyniki planu wykonania i generuj wnioski',
            cs: 'Analyzujte výsledky plánu provádění a generujte poznatky',
            sv: 'Analysera resultat av exekveringsplanen och generera insikter',
            da: 'Analyser resultater af eksekveringsplanen og generer indsigter',
            no: 'Analyser resultater av utførelsesplanen og generer innsikt',
            fi: 'Analysoi suoritussuunnitelman tuloksia ja luo oivalluksia',
            el: 'Ανάλυση αποτελεσμάτων σχεδίου εκτέλεσης και δημιουργία πληροφοριών',
            hu: 'Végrehajtási terv eredményeinek elemzése és betekintések generálása',
            ro: 'Analizați rezultatele planului de execuție și generați perspective',
            uk: 'Аналіз результатів плану виконання та генерація висновків',
            vi: 'Phân tích kết quả kế hoạch thực thi và tạo thông tin chi tiết',
            th: 'วิเคราะห์ผลลัพธ์ของแผนการดำเนินงานและสร้างข้อมูลเชิงลึก',
            id: 'Analisis hasil rencana eksekusi dan hasilkan wawasan'
        },
        status: 'pending',
        stepNum: ''
    },

    render(node, helpers) {
        const el = document.createElement('div');
        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-num">${helpers.escapeHtml(node.stepNum)}</span>
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body">${helpers.escapeHtml(node.tool)}</div>
            <div class="ng-node-progress"></div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
        `;
        return el;
    },

    getDragHandle(el) { return el.querySelector('.ng-node-header'); }
});
