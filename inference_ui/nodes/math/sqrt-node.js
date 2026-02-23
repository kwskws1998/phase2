// ============================================
// Sqrt Node Type (Math category)
// ============================================

NodeRegistry.register('math_sqrt', {
    label: 'Sqrt',
    category: 'Math',

    ports: [
        { name: 'value', dir: 'in', type: 'numeric', label: 'Value', defaultValue: 4 },
        { name: 'out', dir: 'out', type: 'numeric', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Sqrt',
        status: 'pending',
        portValues: { value: 4 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'value', label: 'Value', type: 'numeric', defaultValue: 4 }
        ], 'numeric');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
