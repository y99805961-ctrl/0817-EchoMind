import { HeartFilled, MessageOutlined, MonitorOutlined, ReadOutlined } from "@ant-design/icons";
import { Layout, Menu, Tag } from "antd";
import { useEffect } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { useAppStore } from "../../stores/appStore";

const navItems = [
  { key: "/", icon: <MessageOutlined />, label: <Link to="/">Chat</Link> },
  { key: "/knowledge", icon: <ReadOutlined />, label: <Link to="/knowledge">Knowledge</Link> },
  { key: "/monitor", icon: <MonitorOutlined />, label: <Link to="/monitor">Monitor</Link> },
];

export function AppShell() {
  const location = useLocation();
  const { health, healthError, healthLoading, checkHealth } = useAppStore();

  useEffect(() => {
    void checkHealth();
    const timer = window.setInterval(() => void checkHealth(), 30_000);
    return () => window.clearInterval(timer);
  }, [checkHealth]);

  const online = health?.status === "ok";
  return (
    <Layout className="app-shell">
      <header className="topbar">
        <Link to="/" className="brand-lockup" aria-label="EchoMind Chat">
          <span className="brand-bear">ʕ</span>
          <span><strong>EchoMind</strong><small>智能客服 Agent</small></span>
        </Link>
        <nav className="top-nav" aria-label="Primary navigation">
          {navItems.map((item) => <Link className={location.pathname === item.key ? "active" : ""} key={item.key} to={item.key}>{item.key === "/" ? "Chat" : item.key.slice(1).replace(/^./, (value) => value.toUpperCase())}</Link>)}
        </nav>
        <Tag className={`health-indicator ${online ? "online" : "offline"}`} data-testid="health-indicator">
          <HeartFilled /> {healthLoading ? "Checking…" : online ? "Online" : "Offline"}
        </Tag>
        {healthError && <span className="health-error" title={healthError}>API unavailable</span>}
      </header>
      <Layout className="content-layout">
        <aside className="route-rail" aria-label="Workspace navigation">
          <Menu mode="inline" selectedKeys={[location.pathname]} items={navItems} />
        </aside>
        <main className="page-content"><Outlet /></main>
      </Layout>
    </Layout>
  );
}
