// ============================================
// Protocol Builder Tool Node
// ============================================

NodeRegistry.register('protocol', {
    label: 'Protocol Builder',
    category: 'Tool',
    allowRef: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' },
        { name: 'out', dir: 'out', type: 'any' }
    ],

    defaultConfig: {
        title: 'Protocol Builder',
        tool: 'protocol_builder',
        menuTag: {
            en: 'Protocol', ko: '프로토콜', ja: 'プロトコル', zh: '方案',
            fr: 'Protocole', de: 'Protokoll', es: 'Protocolo', it: 'Protocollo',
            pt: 'Protocolo', nl: 'Protocol', ru: 'Протокол', ar: 'بروتوكول',
            hi: 'प्रोटोकॉल', tr: 'Protokol', pl: 'Protokół', cs: 'Protokol',
            sv: 'Protokoll', da: 'Protokol', no: 'Protokoll', fi: 'Protokolla',
            el: 'Πρωτόκολλο', hu: 'Protokoll', ro: 'Protocol', uk: 'Протокол',
            vi: 'Quy trình', th: 'โปรโตคอล', id: 'Protokol'
        },
        description: {
            en: 'Build experimental protocols step by step',
            ko: '실험 프로토콜을 단계별로 작성',
            ja: '実験プロトコルをステップごとに作成',
            zh: '逐步构建实验方案',
            fr: 'Construire des protocoles expérimentaux étape par étape',
            de: 'Experimentelle Protokolle Schritt für Schritt erstellen',
            es: 'Construir protocolos experimentales paso a paso',
            it: 'Costruire protocolli sperimentali passo dopo passo',
            pt: 'Construir protocolos experimentais passo a passo',
            nl: 'Experimentele protocollen stap voor stap opbouwen',
            ru: 'Пошаговое построение экспериментальных протоколов',
            ar: 'بناء بروتوكولات تجريبية خطوة بخطوة',
            hi: 'प्रयोगात्मक प्रोटोकॉल चरण दर चरण बनाएं',
            tr: 'Deneysel protokolleri adım adım oluşturun',
            pl: 'Tworzenie protokołów eksperymentalnych krok po kroku',
            cs: 'Sestavte experimentální protokoly krok za krokem',
            sv: 'Bygg experimentella protokoll steg för steg',
            da: 'Opbyg eksperimentelle protokoller trin for trin',
            no: 'Bygg eksperimentelle protokoller steg for steg',
            fi: 'Rakenna kokeellisia protokollia vaihe vaiheelta',
            el: 'Δημιουργία πειραματικών πρωτοκόλλων βήμα προς βήμα',
            hu: 'Kísérleti protokollok lépésről lépésre történő összeállítása',
            ro: 'Construiți protocoale experimentale pas cu pas',
            uk: 'Покрокова побудова експериментальних протоколів',
            vi: 'Xây dựng quy trình thí nghiệm từng bước',
            th: 'สร้างโปรโตคอลการทดลองทีละขั้นตอน',
            id: 'Bangun protokol eksperimental langkah demi langkah'
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
