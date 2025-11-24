import React, { useState, useCallback } from 'react';
import ReactFlow, {
    MiniMap,
    Controls,
    Background,
    useNodesState,
    useEdgesState,
    addEdge,
    Connection,
    Edge,
    Node,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { WidgetCard } from './WidgetCard';

const initialNodes: Node[] = [
    { id: '1', position: { x: 0, y: 0 }, data: { label: 'Investment Thesis' }, type: 'input' },
    { id: '2', position: { x: 0, y: 100 }, data: { label: 'Macro Factors' } },
];
const initialEdges: Edge[] = [{ id: 'e1-2', source: '1', target: '2' }];

export const MindMap: React.FC = () => {
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
    const [newNodeLabel, setNewNodeLabel] = useState('');

    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge(params, eds)),
        [setEdges]
    );

    const addNode = () => {
        if (!newNodeLabel.trim()) return;
        const id = `${nodes.length + 1}`;
        const newNode: Node = {
            id,
            position: { x: Math.random() * 200, y: Math.random() * 200 },
            data: { label: newNodeLabel },
        };
        setNodes((nds) => nds.concat(newNode));
        setNewNodeLabel('');
    };

    return (
        <WidgetCard title="INVESTOR MIND MAP" tooltip="Visually map your investment thesis and connect ideas.">
            <div className="h-[400px] w-full bg-slate-900/50 rounded border border-slate-800 relative">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    fitView
                >
                    <Controls className="bg-slate-800 border-slate-700 fill-slate-200" />
                    <MiniMap className="bg-slate-800 border-slate-700" nodeColor="#64748b" />
                    <Background color="#334155" gap={16} />
                </ReactFlow>

                <div className="absolute top-4 left-4 z-10 flex gap-2">
                    <input
                        value={newNodeLabel}
                        onChange={(e) => setNewNodeLabel(e.target.value)}
                        placeholder="New Node Label"
                        className="bg-gray-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 w-32"
                    />
                    <button
                        onClick={addNode}
                        className="bg-blue-900/50 hover:bg-blue-900/70 text-blue-100 px-2 py-1 rounded text-xs border border-blue-800/50"
                    >
                        Add Node
                    </button>
                </div>
            </div>
        </WidgetCard>
    );
};
