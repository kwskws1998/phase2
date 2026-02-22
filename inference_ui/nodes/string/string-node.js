// ============================================
// String Node Type (Input category)
// ============================================

NodeRegistry.register('string', {
    label: 'String',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'string', label: 'Value' }
    ],

    defaultConfig: {
        title: 'String',
        status: 'completed',
        portValues: { out: '' }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const val = (node.portValues && node.portValues.out !== undefined) ? node.portValues.out : '';

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-input-node-body">
                <textarea class="ng-input-node-field ng-port-default ng-interactive" rows="1"
                          placeholder="Enter text..." data-port-ref="out">${helpers.escapeHtml(String(val))}</textarea>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="string" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    afterRender(el, node, helpers) {
        const ta = el.querySelector('textarea');
        if (!ta) return;
        const autoResize = () => {
            ta.style.height = 'auto';
            ta.style.height = ta.scrollHeight + 'px';
            helpers.updateConnections(node);
        };
        ta.addEventListener('input', autoResize);

        let lastWidth = 0;
        const ro = new ResizeObserver(entries => {
            const entry = entries[0];
            if (entry) {
                const w = entry.contentRect.width;
                if (w > 0 && w !== lastWidth) {
                    lastWidth = w;
                    requestAnimationFrame(autoResize);
                }
            }
        });
        ro.observe(el);

        requestAnimationFrame(() => requestAnimationFrame(autoResize));
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
