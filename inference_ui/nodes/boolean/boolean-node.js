// ============================================
// Boolean Node Type (Input category)
// ============================================

NodeRegistry.register('boolean_value', {
    label: 'Boolean',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'boolean', label: 'Value' }
    ],

    defaultConfig: {
        title: 'Boolean',
        status: 'completed',
        portValues: { out: false }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const val = (node.portValues && node.portValues.out !== undefined) ? node.portValues.out : false;

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-input-node-body">
                <div class="ng-boolean-toggle">
                    <input class="ng-interactive" type="checkbox" data-port-ref="out" ${val ? 'checked' : ''}>
                    <span class="ng-boolean-label">${val ? 'true' : 'false'}</span>
                </div>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="boolean" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    afterRender(el, node, helpers) {
        const cb = el.querySelector('input[type="checkbox"]');
        const label = el.querySelector('.ng-boolean-label');
        if (!cb) return;
        cb.addEventListener('change', () => {
            const n = helpers.getNode();
            if (!n) return;
            if (!n.portValues) n.portValues = {};
            n.portValues.out = cb.checked;
            if (label) label.textContent = cb.checked ? 'true' : 'false';
        });
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
