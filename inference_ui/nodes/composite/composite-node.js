// ============================================
// Composite Node Type (General category)
// Combines image + text prompt for multimodal inference
// ============================================

NodeRegistry.register('composite', {
    label: 'Composite',
    category: 'General',

    ports: [
        { name: 'image', dir: 'in', type: 'image', label: 'Image' },
        { name: 'prompt', dir: 'in', type: 'string', label: 'Prompt' },
        { name: 'out', dir: 'out', type: 'any', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Composite',
        tool: 'view_image',
        description: 'Analyze image with text prompt via vision encoder',
        status: 'pending'
    },

    render(node, helpers) {
        const el = document.createElement('div');

        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="image" data-port-dir="in" data-port-type="image" data-node-id="${node.id}"></div>
                <div class="ng-port ng-port-in" data-port-name="prompt" data-port-dir="in" data-port-type="string" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body">Image + Text &rarr; Vision</div>
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
