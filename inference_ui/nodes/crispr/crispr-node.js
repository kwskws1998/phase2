// ============================================
// CRISPR Designer Tool Node
// ============================================

NodeRegistry.register('crispr', {
    label: 'CRISPR Designer',
    category: 'Tool',
    allowRef: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' },
        { name: 'out', dir: 'out', type: 'any' }
    ],

    defaultConfig: {
        title: 'CRISPR Designer',
        tool: 'crispr_designer',
        menuTag: {
            en: 'Gene Editing', ko: '유전자 편집', ja: '遺伝子編集', zh: '基因编辑',
            fr: 'Édition génique', de: 'Genbearbeitung', es: 'Edición genética', it: 'Editing genetico',
            pt: 'Edição genética', nl: 'Genbewerking', ru: 'Редактирование генов', ar: 'تحرير الجينات',
            hi: 'जीन संपादन', tr: 'Gen Düzenleme', pl: 'Edycja genów', cs: 'Editace genů',
            sv: 'Genredigering', da: 'Genredigering', no: 'Genredigering', fi: 'Geenimuokkaus',
            el: 'Επεξεργασία γονιδίων', hu: 'Génszerkesztés', ro: 'Editare genetică', uk: 'Редагування генів',
            vi: 'Chỉnh sửa gen', th: 'การแก้ไขยีน', id: 'Penyuntingan Gen'
        },
        description: {
            en: 'Design CRISPR guide RNAs for gene editing experiments',
            ko: '유전자 편집 실험을 위한 CRISPR 가이드 RNA 설계',
            ja: '遺伝子編集実験のためのCRISPRガイドRNAを設計',
            zh: '为基因编辑实验设计CRISPR引导RNA',
            fr: 'Concevoir des ARN guides CRISPR pour des expériences d\'édition génique',
            de: 'CRISPR-Guide-RNAs für Genbearbeitungsexperimente entwerfen',
            es: 'Diseñar ARN guía CRISPR para experimentos de edición genética',
            it: 'Progettare RNA guida CRISPR per esperimenti di editing genetico',
            pt: 'Projetar RNAs guia CRISPR para experimentos de edição genética',
            nl: 'CRISPR-gids-RNA\'s ontwerpen voor genbewerkingsexperimenten',
            ru: 'Проектирование направляющих РНК CRISPR для экспериментов по редактированию генов',
            ar: 'تصميم الحمض النووي الريبوزي الموجه لـ CRISPR لتجارب تحرير الجينات',
            hi: 'जीन संपादन प्रयोगों के लिए CRISPR गाइड RNA डिज़ाइन करें',
            tr: 'Gen düzenleme deneyleri için CRISPR kılavuz RNA\'ları tasarlayın',
            pl: 'Projektowanie prowadzących RNA CRISPR do eksperymentów edycji genów',
            cs: 'Navrhněte vodící RNA CRISPR pro experimenty s editací genů',
            sv: 'Designa CRISPR-guide-RNA för genredigeringsexperiment',
            da: 'Design CRISPR-guide-RNA til genredigeringseksperimenter',
            no: 'Design CRISPR-guide-RNA for genredigeringseksperimenter',
            fi: 'Suunnittele CRISPR-opas-RNA:t geenimuokkauskokeisiin',
            el: 'Σχεδιασμός οδηγών RNA CRISPR για πειράματα επεξεργασίας γονιδίων',
            hu: 'CRISPR vezető RNS-ek tervezése génszerkesztési kísérletekhez',
            ro: 'Proiectarea ARN-urilor ghid CRISPR pentru experimente de editare genetică',
            uk: 'Проектування направляючих РНК CRISPR для експериментів з редагування генів',
            vi: 'Thiết kế RNA hướng dẫn CRISPR cho thí nghiệm chỉnh sửa gen',
            th: 'ออกแบบ RNA นำทาง CRISPR สำหรับการทดลองแก้ไขยีน',
            id: 'Rancang RNA pemandu CRISPR untuk eksperimen penyuntingan gen'
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
