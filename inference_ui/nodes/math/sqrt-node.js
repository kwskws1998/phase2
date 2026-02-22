// ============================================
// Sqrt Node Type (Math category)
// ============================================

NodeRegistry.register('math_sqrt', {
    label: 'Sqrt',
    category: 'Math',

    ports: [
        { name: 'value', dir: 'in', type: 'float', label: 'Value', defaultValue: 4 },
        { name: 'out', dir: 'out', type: 'float', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Sqrt',
        status: 'pending',
        portValues: { value: 4 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'value', label: 'Value', type: 'float', defaultValue: 4 }
        ]);
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
