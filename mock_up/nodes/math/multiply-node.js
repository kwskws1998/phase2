// ============================================
// Multiply Node Type (Math category)
// ============================================

NodeRegistry.register('math_multiply', {
    label: 'Multiply',
    category: 'Math',

    ports: [
        { name: 'a', dir: 'in', type: 'float', label: 'A', defaultValue: 1 },
        { name: 'b', dir: 'in', type: 'float', label: 'B', defaultValue: 1 },
        { name: 'out', dir: 'out', type: 'float', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Multiply',
        status: 'pending',
        portValues: { a: 1, b: 1 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'a', label: 'A', type: 'float', defaultValue: 1 },
            { name: 'b', label: 'B', type: 'float', defaultValue: 1 }
        ]);
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
