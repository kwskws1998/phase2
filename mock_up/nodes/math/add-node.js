// ============================================
// Add Node Type (Math category)
// ============================================

NodeRegistry.register('math_add', {
    label: 'Add',
    category: 'Math',

    ports: [
        { name: 'a', dir: 'in', type: 'float', label: 'A', defaultValue: 0 },
        { name: 'b', dir: 'in', type: 'float', label: 'B', defaultValue: 0 },
        { name: 'out', dir: 'out', type: 'float', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Add',
        status: 'pending',
        portValues: { a: 0, b: 0 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'a', label: 'A', type: 'float', defaultValue: 0 },
            { name: 'b', label: 'B', type: 'float', defaultValue: 0 }
        ]);
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
