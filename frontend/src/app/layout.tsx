import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SyncNet — Coordinated Bot-Network Detection & Reactive Throttling',
  description: 'Research/decision-support prototype for GNN cluster coordination detection and reactive state machine simulation.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0a0e17] text-slate-100 min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
