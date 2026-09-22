import { HugpyConsole } from "./src/console";
import { styles } from "./src/console";

export default function MediaIntelligenceConsole() {
  return (
    <main className={styles.mediaMain}>
      <HugpyConsole />
    </main>
  );
}
