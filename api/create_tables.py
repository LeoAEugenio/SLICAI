from api.database import Base, engine
from api.models import Analysis


def main():
    Base.metadata.create_all(bind=engine)

    print("Tabelas do SLICAI criadas com sucesso.")


if __name__ == "__main__":
    main()