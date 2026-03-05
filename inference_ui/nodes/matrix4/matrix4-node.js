// ============================================
// Matrix 4x4 Node Type (Input category)
// ============================================

NodeRegistry.register('matrix4', {
    label: 'Matrix 4x4',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'matrix', label: 'Matrix' }
    ],

    defaultConfig: {
        title: 'Matrix 4x4',
        menuTag: { en: 'Matrix', ko: '행렬', ja: '行列', zh: '矩阵', fr: 'Matrice', de: 'Matrix', es: 'Matriz', it: 'Matrice', pt: 'Matriz', nl: 'Matrix', ru: 'Матрица', ar: 'مصفوفة', hi: 'मैट्रिक्स', tr: 'Matris', pl: 'Macierz', cs: 'Matice', sv: 'Matris', da: 'Matrix', no: 'Matrise', fi: 'Matriisi', el: 'Μήτρα', hu: 'Mátrix', ro: 'Matrice', uk: 'Матриця', vi: 'Ma trận', th: 'เมทริกซ์', id: 'Matriks' },
        description: { en: '4x4 matrix input', ko: '4x4 행렬 입력', ja: '4x4 行列入力', zh: '4x4 矩阵输入', fr: 'Entrée matrice 4x4', de: '4x4-Matrixeingabe', es: 'Entrada de matriz 4x4', it: 'Input matrice 4x4', pt: 'Entrada de matriz 4x4', nl: '4x4 matrix-invoer', ru: 'Ввод матрицы 4x4', ar: 'إدخال مصفوفة 4x4', hi: '4x4 मैट्रिक्स इनपुट', tr: '4x4 matris girişi', pl: 'Wejście macierzy 4x4', cs: 'Vstup matice 4x4', sv: '4x4 matrisinmatning', da: '4x4 matrix-input', no: '4x4 matriseinndata', fi: '4x4-matriisin syöttö', el: 'Είσοδος μήτρας 4x4', hu: '4x4 mátrix bevitel', ro: 'Intrare matrice 4x4', uk: 'Введення матриці 4x4', vi: 'Nhập ma trận 4x4', th: 'อินพุตเมทริกซ์ 4x4', id: 'Input matriks 4x4' },
        status: 'completed',
        portValues: { out: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const identity = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
        const vals = (node.portValues && Array.isArray(node.portValues.out))
            ? node.portValues.out : identity;

        const cells = [];
        for (let i = 0; i < 16; i++) {
            const v = vals[i] !== undefined ? vals[i] : identity[i];
            cells.push(`<input class="ng-matrix-cell ng-interactive" type="number" step="any" value="${v}" data-cell="${i}">`);
        }

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-matrix-body">
                <div class="ng-matrix-grid ng-matrix-4x4">
                    ${cells.join('\n')}
                </div>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="matrix" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    afterRender(el, node, helpers) {
        const updateMatrix = () => {
            const n = helpers.getNode();
            if (!n) return;
            if (!n.portValues) n.portValues = {};
            const cells = el.querySelectorAll('.ng-matrix-cell');
            n.portValues.out = Array.from(cells).map(c => parseFloat(c.value) || 0);
        };
        el.querySelectorAll('.ng-matrix-cell').forEach(cell => {
            cell.addEventListener('change', updateMatrix);
            cell.addEventListener('input', updateMatrix);
        });
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
