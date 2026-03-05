// ============================================
// Step Node Type
// ============================================

NodeRegistry.register('step', {
    label: 'Step',
    category: 'General',
    allowRef: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' },
        { name: 'out', dir: 'out', type: 'any' }
    ],

    defaultConfig: {
        title: 'Step',
        tool: '',
        menuTag: { en: 'Execution', ko: '실행', ja: '実行', zh: '执行', fr: 'Exécution', de: 'Ausführung', es: 'Ejecución', it: 'Esecuzione', pt: 'Execução', nl: 'Uitvoering', ru: 'Выполнение', ar: 'تنفيذ', hi: 'निष्पादन', tr: 'Yürütme', pl: 'Wykonanie', cs: 'Spuštění', sv: 'Körning', da: 'Udførelse', no: 'Utførelse', fi: 'Suoritus', el: 'Εκτέλεση', hu: 'Végrehajtás', ro: 'Execuție', uk: 'Виконання', vi: 'Thực thi', th: 'การดำเนินการ', id: 'Eksekusi' },
        description: { en: 'General-purpose execution step for running tools and actions', ko: '도구와 액션을 실행하는 범용 실행 단계', ja: 'ツールとアクションを実行する汎用実行ステップ', zh: '用于运行工具和操作的通用执行步骤', fr: 'Étape d\'exécution polyvalente pour lancer des outils et des actions', de: 'Allgemeiner Ausführungsschritt zum Ausführen von Werkzeugen und Aktionen', es: 'Paso de ejecución de propósito general para ejecutar herramientas y acciones', it: 'Passaggio di esecuzione generico per eseguire strumenti e azioni', pt: 'Etapa de execução de uso geral para executar ferramentas e ações', nl: 'Algemene uitvoeringsstap voor het uitvoeren van tools en acties', ru: 'Универсальный шаг выполнения для запуска инструментов и действий', ar: 'خطوة تنفيذ عامة لتشغيل الأدوات والإجراءات', hi: 'टूल और क्रियाओं को चलाने के लिए सामान्य-उद्देश्य निष्पादन चरण', tr: 'Araçları ve eylemleri çalıştırmak için genel amaçlı yürütme adımı', pl: 'Uniwersalny krok wykonania do uruchamiania narzędzi i akcji', cs: 'Univerzální krok pro spouštění nástrojů a akcí', sv: 'Allmänt körningssteg för att köra verktyg och åtgärder', da: 'Generelt udførelsestrin til at køre værktøjer og handlinger', no: 'Generelt utførelsestrinn for å kjøre verktøy og handlinger', fi: 'Yleiskäyttöinen suoritusvaihe työkalujen ja toimintojen ajamiseen', el: 'Βήμα εκτέλεσης γενικού σκοπού για εκτέλεση εργαλείων και ενεργειών', hu: 'Általános célú végrehajtási lépés eszközök és műveletek futtatásához', ro: 'Pas de execuție de uz general pentru rularea uneltelor și acțiunilor', uk: 'Універсальний крок виконання для запуску інструментів та дій', vi: 'Bước thực thi đa năng để chạy các công cụ và hành động', th: 'ขั้นตอนการดำเนินการทั่วไปสำหรับเรียกใช้เครื่องมือและการกระทำ', id: 'Langkah eksekusi umum untuk menjalankan alat dan tindakan' },
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
                <span class="ng-node-num">${node.stepNum}</span>
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

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
