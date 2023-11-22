import { Construct } from 'constructs';
import { config } from './config';
import {
  ApplicationDynamoDBTable,
  ApplicationDynamoDBTableCapacityMode,
} from '@pocket-tools/terraform-modules';

export class DynamoDB extends Construct {
  public readonly indexStoreTable: ApplicationDynamoDBTable;
  public readonly documentStoreTable: ApplicationDynamoDBTable;

  constructor(scope: Construct, name: string) {
    super(scope, name);
    this.indexStoreTable = this.setupIndexStoreTable();
    this.documentStoreTable = this.setupDocumentStoreTable();
  }

  /**
   * Sets up the dynamodb table for the index store
   * @private
   */
  private setupIndexStoreTable() {
    return new ApplicationDynamoDBTable(this, `index_store`, {
      tags: config.tags,
      prefix: `${config.shortName}-${config.environment}-IndexStore`,
      capacityMode: ApplicationDynamoDBTableCapacityMode.ON_DEMAND,
      tableConfig: {
        streamEnabled: false,
        hashKey: 'collection',
        rangeKey: 'key',
        attribute: [
          {
            name: 'collection',
            type: 'S',
          },
          {
            name: 'key',
            type: 'S',
          },
        ]
      },
    });
  }

    /**
   * Sets up the dynamodb table for the index store
   * @private
   */
  private setupDocumentStoreTable() {
    return new ApplicationDynamoDBTable(this, `document_store`, {
      tags: config.tags,
      prefix: `${config.shortName}-${config.environment}-DocumentStore`,
      capacityMode: ApplicationDynamoDBTableCapacityMode.ON_DEMAND,
      tableConfig: {
        streamEnabled: false,
        hashKey: 'collection',
        rangeKey: 'key',
        attribute: [
          {
            name: 'collection',
            type: 'S',
          },
          {
            name: 'key',
            type: 'S',
          },
        ]
      },
    });
  }





}