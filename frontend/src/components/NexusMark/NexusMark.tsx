interface NexusMarkProps {
  small?: boolean;
}

export function NexusMark({ small = false }: NexusMarkProps) {
  return (
    <span className={`nexus-mark ${small ? "nexus-mark-small" : ""}`} aria-hidden="true">
      <span className="nexus-link nexus-link-one" />
      <span className="nexus-link nexus-link-two" />
      <span className="nexus-link nexus-link-three" />
      <span className="nexus-node nexus-node-one" />
      <span className="nexus-node nexus-node-two" />
      <span className="nexus-node nexus-node-three" />
      <span className="nexus-core" />
    </span>
  );
}
