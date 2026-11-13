"""本地初始化账号命令，不提供公开的 HR 注册接口。"""
import argparse
from getpass import getpass
from qiye_rag.run_time.database import create_sessions
from qiye_rag.run_time.auth import create_user


def main() -> None:
    """创建数据库表与一个用户，密码通过终端隐藏输入。"""
    parser = argparse.ArgumentParser(description="初始化 v4 账号")
    parser.add_argument("username")
    parser.add_argument("--role", choices=["user", "hr"], default="user")
    args = parser.parse_args()
    password = getpass("密码（至少8位）: ")
    if password != getpass("确认密码: "):
        raise ValueError("两次密码不一致")
    sessions = create_sessions()
    with sessions() as db:
        create_user(db, args.username, password, args.role)
    print("账号已创建")


if __name__ == "__main__":
    main()
