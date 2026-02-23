// ============================================
// Matrix 2x2 Node Type (Input category)
// ============================================

NodeRegistry.register('matrix2', {
    label: 'Matrix 2x2',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'matrix', label: 'Matrix' }
    ],

    defaultConfig: {
        title: 'Matrix 2x2',
        status: 'completed',
        portValues: { out: [1,0, 0,1] }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const identity = [1,0, 0,1];
        const vals = (node.portValues && Array.isArray(node.portValues.out))
            ? node.portValues.out : identity;

        const cells = [];
        for (let i = 0; i < 4; i++) {
            const v = vals[i] !== undefined ? vals[i] : identity[i];
            cells.push(`<input class="ng-matrix-cell ng-interactive" type="number" step="any" value="${v}" data-cell="${i}">`);
        }

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-matrix-body">
                <div class="ng-matrix-grid ng-matrix-2x2">
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
